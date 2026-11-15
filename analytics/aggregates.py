"""Per-dish waste summaries computed from stored plate events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PlateObservation:
    """One food region on one returned plate, as analytics sees it."""

    hall: str
    meal: str
    observed_at: datetime
    dish_name: str | None
    grams: float
    percent_left: float | None


@dataclass(frozen=True, slots=True)
class DishSummary:
    """Waste numbers for one dish over a period."""

    dish_name: str
    plates_seen: int
    total_kg: float
    mean_percent_left: float | None


def summarize_dishes(observations: Sequence[PlateObservation]) -> list[DishSummary]:
    """Group observations by dish and total them, sorted by kilograms wasted.

    Regions matched to no dish (``dish_name`` None) are excluded here and
    counted by the review queue instead.
    """
    raise NotImplementedError("hour 5-9")


def flag_high_waste(summaries: Sequence[DishSummary], threshold_percent: float) -> list[str]:
    """Names of dishes whose mean percent left is above ``threshold_percent``."""
    raise NotImplementedError("hour 5-9")
