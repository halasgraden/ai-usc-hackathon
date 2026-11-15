"""Map each menu dish to one coarse food category.

Decision order for a dish: manual override, then a category it already carries
(synthetic composite dishes), then the cache, then the keyword rules. Rule
decisions are cached by dish key so each new dish is categorized once; the
cache is invalidated automatically when the rules or overrides change.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.types import FoodCategory
from menu.cleaner import dish_key
from menu.models import DailyMenu, ItemRole, MenuItem

DEFAULT_RULES_PATH = Path(__file__).parent / "data" / "category_rules.yaml"
DEFAULT_OVERRIDES_PATH = Path(__file__).parent / "data" / "overrides.yaml"

_ANIMAL = frozenset({FoodCategory.POULTRY, FoodCategory.RED_MEAT, FoodCategory.FISH_SEAFOOD})


@dataclass(frozen=True, slots=True)
class CategoryRule:
    category: FoodCategory
    pattern: re.Pattern[str]
    stations: frozenset[str]

    def matches(self, item: MenuItem) -> bool:
        if self.stations and not self.stations.intersection(item.stations):
            return False
        return self.pattern.search(item.key) is not None


class CategoryCache:
    """JSON file of rule decisions, keyed by dish key.

    The file records a fingerprint of the rules that produced it. Opening it
    with a different fingerprint starts empty, so stale decisions never
    outlive a rules change.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fingerprint = ""
        self._entries: dict[str, str] = {}
        self._dirty = False

    def open(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint
        self._entries = {}
        if not self.path.exists():
            return
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError:
            return
        if isinstance(stored, dict) and stored.get("fingerprint") == fingerprint:
            self._entries = dict(stored.get("entries", {}))

    def get(self, key: str) -> FoodCategory | None:
        value = self._entries.get(key)
        try:
            return FoodCategory(value) if value is not None else None
        except ValueError:
            return None

    def put(self, key: str, category: FoodCategory) -> None:
        if self._entries.get(key) != category.value:
            self._entries[key] = category.value
            self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": self._fingerprint, "entries": self._entries}
        self.path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        self._dirty = False


class Categorizer:
    """Assigns a ``FoodCategory`` to menu items."""

    def __init__(
        self,
        rules: list[CategoryRule],
        overrides: dict[str, FoodCategory | None] | None = None,
        cache: CategoryCache | None = None,
        fingerprint: str = "",
    ) -> None:
        self.rules = rules
        self.overrides = {dish_key(k): v for k, v in (overrides or {}).items()}
        self.cache = cache
        if cache is not None:
            cache.open(fingerprint)

    @classmethod
    def load(
        cls,
        rules_path: Path = DEFAULT_RULES_PATH,
        overrides_path: Path = DEFAULT_OVERRIDES_PATH,
        cache: CategoryCache | None = None,
    ) -> Categorizer:
        """Build a categorizer from the YAML rule and override files.

        Raises:
            ValueError: a file names a category that does not exist.
        """
        rules_text = rules_path.read_text(encoding="utf-8")
        overrides_text = overrides_path.read_text(encoding="utf-8")
        rules = [
            CategoryRule(
                category=FoodCategory(r["category"]),
                pattern=re.compile(rf"\b(?:{r['pattern']})\b"),
                stations=frozenset(r.get("stations", [])),
            )
            for r in yaml.safe_load(rules_text)["rules"]
        ]
        raw_overrides = (yaml.safe_load(overrides_text) or {}).get("overrides") or {}
        overrides = {
            name: (FoodCategory(value) if value is not None else None)
            for name, value in raw_overrides.items()
        }
        fingerprint = hashlib.sha256((rules_text + overrides_text).encode()).hexdigest()
        return cls(rules, overrides, cache, fingerprint)

    def _apply_rules(self, item: MenuItem) -> FoodCategory | None:
        plant_based = bool({"vegan", "vegetarian"}.intersection(item.preferences))
        vegan = "vegan" in item.preferences
        for rule in self.rules:
            if plant_based and rule.category in _ANIMAL:
                continue
            if vegan and rule.category is FoodCategory.EGGS:
                continue
            if rule.matches(item):
                return rule.category
        return None

    def categorize(self, item: MenuItem) -> tuple[FoodCategory | None, str | None]:
        """Return ``(category, source)`` for one item; ``(None, None)`` if no rule fits.

        ``source`` is "override", "composite", "cache" or "rule".
        """
        if item.key in self.overrides:
            return self.overrides[item.key], "override"
        if item.category is not None:
            return item.category, item.category_source
        if self.cache is not None and (cached := self.cache.get(item.key)) is not None:
            return cached, "cache"
        category = self._apply_rules(item)
        if category is None:
            return None, None
        if self.cache is not None:
            self.cache.put(item.key, category)
        return category, "rule"

    def apply(self, menu: DailyMenu) -> DailyMenu:
        """Return a copy of ``menu`` with dishes and components categorized.

        Condiments and notices are left without a category.
        """
        result = menu.model_copy(deep=True)
        for meal in result.meals:
            for item in meal.items:
                if item.role in (ItemRole.DISH, ItemRole.COMPONENT):
                    item.category, item.category_source = self.categorize(item)
        if self.cache is not None:
            self.cache.save()
        return result
