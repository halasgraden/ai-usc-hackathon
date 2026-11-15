"""Turn the raw API menu into the list of dishes a plate may contain.

For every meal: normalize names, give each entry a role (dish, component,
condiment, notice), merge entries repeated across stations, and add the dishes
the menu implies without listing (an omelette, a deli sandwich). Nothing is
discarded: ``MealMenu.candidates()`` is the filtered view the matcher uses, and
the other roles stay available for the dashboard and for debugging the rules.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.types import FoodCategory
from menu.models import DailyMenu, ItemRole, MealMenu, MenuItem, RawMeal, RawMenu, RawMenuItem

DEFAULT_RULES_PATH = Path(__file__).parent / "data" / "cleaning_rules.yaml"

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(raw: str) -> str:
    """Display form of a name: HTML entities decoded, whitespace collapsed."""
    return _WHITESPACE.sub(" ", html.unescape(raw)).strip()


def dish_key(name: str) -> str:
    """Identity of a dish: ASCII lower case with punctuation collapsed to spaces.

    >>> dish_key("Oat n' Honey  Granola")
    'oat n honey granola'
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return _NON_ALNUM.sub(" ", ascii_name.casefold()).strip()


def _phrase_pattern(phrases: list[str]) -> re.Pattern[str] | None:
    if not phrases:
        return None
    alternatives = "|".join(re.escape(dish_key(p)) for p in phrases)
    return re.compile(rf"\b(?:{alternatives})\b")


@dataclass(frozen=True, slots=True)
class CompositeRule:
    """A dish implied by other entries (see ``composites`` in the rules file)."""

    name: str
    category: FoodCategory
    pattern: re.Pattern[str]
    min_matches: int


@dataclass(frozen=True, slots=True)
class CleaningRules:
    """Compiled contents of ``cleaning_rules.yaml``."""

    station_aliases: dict[str, str]
    notice_patterns: tuple[re.Pattern[str], ...]
    condiment_anywhere: re.Pattern[str] | None
    condiment_suffixes: frozenset[str]
    component_prefixes: frozenset[str]
    component_exact: frozenset[str]
    component_anywhere: re.Pattern[str] | None
    composites: tuple[CompositeRule, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CleaningRules:
        condiments = data.get("condiments", {})
        components = data.get("components", {})
        return cls(
            station_aliases=dict(data.get("station_aliases", {})),
            notice_patterns=tuple(re.compile(p) for p in data.get("notice_patterns", [])),
            condiment_anywhere=_phrase_pattern(condiments.get("anywhere", [])),
            condiment_suffixes=frozenset(dish_key(s) for s in condiments.get("suffix", [])),
            component_prefixes=frozenset(dish_key(s) for s in components.get("prefix", [])),
            component_exact=frozenset(dish_key(s) for s in components.get("exact", [])),
            component_anywhere=_phrase_pattern(components.get("anywhere", [])),
            composites=tuple(
                CompositeRule(
                    name=c["name"],
                    category=FoodCategory(c["category"]),
                    pattern=re.compile(c["pattern"]),
                    min_matches=int(c.get("min_matches", 1)),
                )
                for c in data.get("composites", [])
            ),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_RULES_PATH) -> CleaningRules:
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def classify_role(entry: RawMenuItem, rules: CleaningRules) -> ItemRole:
    """Decide what a raw entry is. Checks run notice, condiment, component, dish."""
    name = normalize_name(entry.item)
    key = dish_key(name)
    words = key.split()
    if not words:
        return ItemRole.NOTICE

    untagged = not (entry.allergens or entry.preferences or entry.dietary_preferences)
    if untagged and name == name.upper() and any(ch.isalpha() for ch in name):
        return ItemRole.NOTICE
    if any(p.search(key) for p in rules.notice_patterns):
        return ItemRole.NOTICE

    if rules.condiment_anywhere and rules.condiment_anywhere.search(key):
        return ItemRole.CONDIMENT
    is_plain_name = "with" not in words and "in" not in words
    if is_plain_name and words[-1] in rules.condiment_suffixes:
        return ItemRole.CONDIMENT

    if words[0] in rules.component_prefixes or key in rules.component_exact:
        return ItemRole.COMPONENT
    if rules.component_anywhere and rules.component_anywhere.search(key):
        return ItemRole.COMPONENT
    return ItemRole.DISH


def _merge_into(existing: MenuItem, station: str, entry: RawMenuItem) -> None:
    if station not in existing.stations:
        existing.stations.append(station)
    for tag in entry.allergens:
        if tag not in existing.allergens:
            existing.allergens.append(tag)
    for tag in entry.preferences:
        if tag not in existing.preferences:
            existing.preferences.append(tag)


def clean_meal(meal: RawMeal, rules: CleaningRules) -> MealMenu:
    """Clean one meal: assign roles, merge repeats across stations, add composites."""
    items: dict[str, MenuItem] = {}
    for raw_station in meal.stations:
        station = normalize_name(raw_station.station)
        station = rules.station_aliases.get(station, station)
        for entry in raw_station.menu:
            name = normalize_name(entry.item)
            key = dish_key(name)
            if not key:
                continue
            if key in items:
                _merge_into(items[key], station, entry)
                continue
            items[key] = MenuItem(
                name=name,
                key=key,
                role=classify_role(entry, rules),
                stations=[station],
                allergens=list(entry.allergens),
                preferences=list(entry.preferences),
            )

    for rule in rules.composites:
        matches = [item for key, item in items.items() if rule.pattern.search(key)]
        key = dish_key(rule.name)
        if len(matches) < rule.min_matches or key in items:
            continue
        items[key] = MenuItem(
            name=rule.name,
            key=key,
            role=ItemRole.DISH,
            stations=[matches[0].stations[0]],
            category=rule.category,
            category_source="composite",
            synthetic=True,
        )
    return MealMenu(name=normalize_name(meal.name), items=list(items.values()))


def clean_menu(raw: RawMenu, rules: CleaningRules, *, hall: str, source: str = "api") -> DailyMenu:
    """Clean a whole day. Meals with no stations (not served that day) are dropped.

    Raises:
        ValueError: the menu's date is not in ``YYYYMMDD`` form.
    """
    day = datetime.strptime(raw.date, "%Y%m%d").date()
    meals = [clean_meal(m, rules) for m in raw.meals if m.stations]
    return DailyMenu(day=day, hall=hall, source=source, meals=meals)
