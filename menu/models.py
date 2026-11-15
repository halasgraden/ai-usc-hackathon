"""Data models for the menu stage.

``Raw*`` models mirror the USC Hospitality JSON exactly as served. ``MenuItem``,
``MealMenu`` and ``DailyMenu`` are what the rest of the pipeline consumes.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.types import FoodCategory


def _as_str_list(value: Any) -> list[str]:
    """The WordPress endpoint may send ``false``/``""``/``null`` for an empty list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


class RawMenuItem(BaseModel):
    """One entry under a station: a dish, a condiment or a notice line."""

    model_config = ConfigDict(extra="ignore")

    item: str
    dietary_preferences: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)

    _lists = field_validator("dietary_preferences", "allergens", "preferences", mode="before")(
        _as_str_list
    )


class RawStation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    station: str
    subtitle: str = ""
    menu: list[RawMenuItem] = Field(default_factory=list)


class RawMeal(BaseModel):
    """A meal period. Meals that are not served that day have no ``stations`` key."""

    model_config = ConfigDict(extra="ignore")

    name: str
    stations: list[RawStation] = Field(default_factory=list)


class RawMenu(BaseModel):
    """Top-level API response for one hall and one day."""

    model_config = ConfigDict(extra="ignore")

    date: str
    location: str = ""
    meals: list[RawMeal] = Field(default_factory=list)

    def entry_count(self, meal: str) -> int:
        """Number of raw entries (dishes, condiments and notices) listed for ``meal``."""
        return sum(len(s.menu) for m in self.meals if m.name == meal for s in m.stations)


class ItemRole(StrEnum):
    """What a menu entry is, from the camera's point of view."""

    DISH = "dish"
    """Measurable on a plate; a matching candidate."""
    COMPONENT = "component"
    """A build-your-own filling or topping (omelette fillings, deli meats). Kept for
    reference, never a candidate: the plate shows the assembled dish instead."""
    CONDIMENT = "condiment"
    """Sauces, dressings, butter, spreads: not measurable on a plate."""
    NOTICE = "notice"
    """Not food: "GRILL ITEMS AVAILABLE AT 11AM"."""


class MenuItem(BaseModel):
    """A cleaned menu entry, merged across the stations that list it."""

    name: str
    key: str
    """Normalized name: the identity used for merging, overrides and caching."""
    role: ItemRole
    stations: list[str]
    allergens: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    category: FoodCategory | None = None
    category_source: str | None = None
    """"override", "composite", "cache" or "rule"; None while uncategorized."""
    synthetic: bool = False
    """True for dishes we add ourselves (e.g. the omelette implied by its fillings)."""


class MealMenu(BaseModel):
    name: str
    items: list[MenuItem]

    def candidates(self) -> list[MenuItem]:
        """The dishes a plate region may be matched to during this meal."""
        return [i for i in self.items if i.role is ItemRole.DISH]

    def uncategorized(self) -> list[MenuItem]:
        """Candidates no rule could place; add them to ``menu/data/overrides.yaml``."""
        return [i for i in self.candidates() if i.category is None]


class DailyMenu(BaseModel):
    """One hall's cleaned menu for one day. Only meals actually served are listed."""

    day: date
    hall: str
    source: str
    """Where the raw menu came from: "api", "cache" or "manual"."""
    meals: list[MealMenu]

    def meal(self, name: str) -> MealMenu | None:
        """Look a meal up by name, case-insensitively. None if not served that day."""
        wanted = name.casefold()
        return next((m for m in self.meals if m.name.casefold() == wanted), None)
