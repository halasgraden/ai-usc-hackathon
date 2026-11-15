"""Per-dish waste summaries computed from stored plate events.

Everything here is pure: it takes observations and returns numbers, so the API,
the seed script and the tests share one definition of every figure.

``percent_left`` is an index (leftover volume against one reference portion per
dish, see ``portion.grams.percent_left``), not a share of what a student took.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import fmean
from typing import Literal

Trend = Literal["up", "down", "flat", "unknown"]

TREND_TOLERANCE_POINTS = 2.0
"""A change in mean percent left smaller than this (in index points) counts as flat."""


@dataclass(frozen=True, slots=True)
class PlateObservation:
    """One food region on one returned plate, as analytics sees it."""

    hall: str
    meal: str
    observed_at: datetime
    dish_name: str | None
    grams: float
    percent_left: float | None
    plate_id: str | None = None
    """Random plate id, used only to count distinct plates. None: count each region."""


@dataclass(frozen=True, slots=True)
class DishSummary:
    """Waste numbers for one dish over a period."""

    dish_name: str
    plates_seen: int
    total_kg: float
    mean_percent_left: float | None


@dataclass(frozen=True, slots=True)
class WeekPoint:
    """Totals for one Monday-to-Sunday week."""

    week_start: date
    kg_wasted: float
    waste_rate_percent: float | None


def summarize_dishes(observations: Sequence[PlateObservation]) -> list[DishSummary]:
    """Group observations by dish and total them, sorted by kilograms wasted.

    Regions matched to no dish (``dish_name`` None) are excluded here and
    counted by the review queue instead. Ties are broken by dish name so the
    order is stable.
    """
    groups: dict[str, list[PlateObservation]] = {}
    for item in observations:
        if item.dish_name is not None:
            groups.setdefault(item.dish_name, []).append(item)

    summaries = []
    for name, items in groups.items():
        plates = {o.plate_id if o.plate_id is not None else f"#{i}" for i, o in enumerate(items)}
        summaries.append(
            DishSummary(
                dish_name=name,
                plates_seen=len(plates),
                total_kg=sum(o.grams for o in items) / 1000.0,
                mean_percent_left=mean_percent_left(items),
            )
        )
    summaries.sort(key=lambda s: (-s.total_kg, s.dish_name))
    return summaries


def flag_high_waste(summaries: Sequence[DishSummary], threshold_percent: float) -> list[str]:
    """Names of dishes whose mean percent left is above ``threshold_percent``.

    Dishes without any percent-left value are never flagged. The order of
    ``summaries`` is kept.
    """
    return [
        s.dish_name
        for s in summaries
        if s.mean_percent_left is not None and s.mean_percent_left > threshold_percent
    ]


def total_kg(observations: Iterable[PlateObservation]) -> float:
    """Kilograms across every region, matched or not (unknown food is still waste)."""
    return sum(o.grams for o in observations) / 1000.0


def mean_percent_left(observations: Iterable[PlateObservation]) -> float | None:
    """Mean percent-left index across matched regions; None if there are none."""
    values = [
        o.percent_left
        for o in observations
        if o.dish_name is not None and o.percent_left is not None
    ]
    return fmean(values) if values else None


def week_start(day: date) -> date:
    """The Monday of the week containing ``day``."""
    return day - timedelta(days=day.weekday())


def weekly_points(observations: Iterable[PlateObservation]) -> list[WeekPoint]:
    """One point per Monday-to-Sunday week that has observations, oldest first.

    Weeks without any observation are left out rather than reported as zero
    waste: no data is not the same as no waste.
    """
    weeks: dict[date, list[PlateObservation]] = {}
    for item in observations:
        weeks.setdefault(week_start(item.observed_at.date()), []).append(item)
    return [
        WeekPoint(start, total_kg(items), mean_percent_left(items))
        for start, items in sorted(weeks.items())
    ]


def trend_direction(
    current: float | None,
    previous: float | None,
    tolerance: float = TREND_TOLERANCE_POINTS,
) -> Trend:
    """Compare two mean percent-left values: ``up`` means more food left over."""
    if current is None or previous is None:
        return "unknown"
    if current - previous >= tolerance:
        return "up"
    if previous - current >= tolerance:
        return "down"
    return "flat"
