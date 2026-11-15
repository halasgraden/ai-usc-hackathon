"""Read endpoints for the staff dashboard and the public weekly report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, col, func, select

from analytics.aggregates import (
    PlateObservation,
    mean_percent_left,
    summarize_dishes,
    total_kg,
    trend_direction,
    week_start,
    weekly_points,
)
from analytics.assumptions import CostFactors
from api.dishes import station_of
from api.models import Database, Plate, PlateEvent
from api.schemas import (
    DataSource,
    DishDetailOut,
    DishesOut,
    DishRow,
    Headline,
    ReportOut,
    TodayOut,
    TrendPoint,
    TrendsOut,
)
from core.types import FoodCategory

TOP_DISHES = 5
DEFAULT_DISHES_DAYS = 7
DEFAULT_DETAIL_DAYS = 28
MAX_RANGE_DAYS = 366
MAX_SAMPLES = 5000

METHODS_NOTE = (
    "A camera over the USC Village dish return photographs each returned plate. A vision "
    "model outlines each food region, estimates its volume from depth, and matches it to a "
    "dish on that day's menu. Grams are volume x a density for the food's category, so every "
    "weight is an estimate with a range. 'Percent left' is an index: leftover volume against "
    "one reference portion per dish, not a share of what a student took (the hall is "
    "self-serve), and it can exceed 100. Regions the system could not match to a dish count "
    "toward kg wasted but not toward any dish or the waste rate, and wait for staff review. "
    "Cost and CO2e are kg wasted x the assumed factors listed with this report, not "
    "measurements. No student, card, device or face information is collected; times are "
    "stored to the nearest 15 minutes."
)
SEEDED_NOTE = " This report is built from simulated (seeded) demo data, not from real plates."


def _start_of(day: date) -> datetime:
    return datetime.combine(day, time.min)


def _r(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _observations(events: Sequence[PlateEvent]) -> list[PlateObservation]:
    return [
        PlateObservation(
            e.hall, e.meal, e.observed_at, e.dish_name, e.grams, e.percent_left, e.plate_id
        )
        for e in events
    ]


def _headline(events: Sequence[PlateEvent], factors: CostFactors) -> Headline:
    observations = _observations(events)
    kg = total_kg(observations)
    return Headline(
        kg_wasted=round(kg, 2),
        waste_rate_percent=_r(mean_percent_left(observations), 1),
        est_cost_usd=round(factors.est_cost_usd(kg), 2),
        est_co2e_kg=round(factors.est_co2e_kg(kg), 2),
    )


def _category(values: Counter[str]) -> FoodCategory | None:
    if not values:
        return None
    try:
        return FoodCategory(values.most_common(1)[0][0])
    except ValueError:
        return None


def _rows(current: Sequence[PlateEvent], previous: Sequence[PlateEvent]) -> list[DishRow]:
    """Dish rows, most kg first; ``trend`` compares mean percent left with ``previous``."""
    before = {s.dish_name: s.mean_percent_left for s in summarize_dishes(_observations(previous))}
    categories: dict[str, Counter[str]] = {}
    for e in current:
        if e.dish_name is not None:
            categories.setdefault(e.dish_name, Counter())[e.category] += 1
    return [
        DishRow(
            dish_name=s.dish_name,
            station=station_of(s.dish_name),
            category=_category(categories.get(s.dish_name, Counter())),
            plates_seen=s.plates_seen,
            total_kg=round(s.total_kg, 3),
            mean_percent_left=_r(s.mean_percent_left, 1),
            trend=trend_direction(s.mean_percent_left, before.get(s.dish_name)),
        )
        for s in summarize_dishes(_observations(current))
    ]


def _check_range(start: date, end: date) -> None:
    if start > end:
        raise HTTPException(422, "start must not be after end")
    if (end - start).days >= MAX_RANGE_DAYS:
        raise HTTPException(422, f"range is limited to {MAX_RANGE_DAYS} days")


class _Store:
    """Queries over one hall's plates. Windows are ``[start, end)`` in hall-local days."""

    def __init__(self, session: Session, hall: str) -> None:
        self.session = session
        self.hall = hall

    def events(
        self, start: date, end: date, meal: str | None = None, dish: str | None = None
    ) -> list[PlateEvent]:
        query = select(PlateEvent).where(
            PlateEvent.hall == self.hall,
            col(PlateEvent.observed_at) >= _start_of(start),
            col(PlateEvent.observed_at) < _start_of(end),
        )
        if meal:
            query = query.where(func.lower(PlateEvent.meal) == meal.strip().lower())
        if dish is not None:
            query = query.where(PlateEvent.dish_name == dish)
        return list(self.session.exec(query.order_by(col(PlateEvent.observed_at))).all())

    def plate_count(self, start: date, end: date, meal: str | None = None) -> int:
        query = (
            select(func.count())
            .select_from(Plate)
            .where(
                Plate.hall == self.hall,
                col(Plate.observed_at) >= _start_of(start),
                col(Plate.observed_at) < _start_of(end),
            )
        )
        if meal:
            query = query.where(func.lower(Plate.meal) == meal.strip().lower())
        return int(self.session.exec(query).one())

    def data_source(self, start: date, end: date) -> DataSource:
        """``seeded`` if any plate in the window is seeded.

        An empty window in a database that holds seeded plates is also labelled
        seeded, so a demo never shows an unlabelled screen.
        """
        in_window = select(Plate.data_source).where(
            Plate.hall == self.hall,
            col(Plate.observed_at) >= _start_of(start),
            col(Plate.observed_at) < _start_of(end),
        )
        if self.session.exec(in_window.where(Plate.data_source == "seeded").limit(1)).first():
            return "seeded"
        if self.session.exec(in_window.limit(1)).first() is None:
            any_seeded = select(Plate.id).where(
                Plate.hall == self.hall, Plate.data_source == "seeded"
            )
            if self.session.exec(any_seeded.limit(1)).first() is not None:
                return "seeded"
        return "live"


def build_router(
    db: Database,
    *,
    hall: str,
    timezone: str,
    factors: CostFactors,
    require_staff: Callable[..., None],
) -> APIRouter:
    router = APIRouter()
    staff = [Depends(require_staff)]

    def today() -> date:
        return datetime.now(ZoneInfo(timezone)).date()

    @router.get("/dashboard/today", dependencies=staff, tags=["dashboard"])
    def dashboard_today(meal: str | None = None, day: date | None = None) -> TodayOut:
        """One day (default: today in the hall's time zone), optionally one meal.

        Dish trends compare with the 7 days before ``day``.
        """
        d = day or today()
        end = d + timedelta(days=1)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = store.events(d, end, meal)
            previous = store.events(d - timedelta(days=7), d, meal)
            return TodayOut(
                hall=hall,
                day=d,
                meal=meal,
                data_source=store.data_source(d, end),
                headline=_headline(events, factors),
                top_dishes=_rows(events, previous)[:TOP_DISHES],
                plates_seen=store.plate_count(d, end, meal),
                regions_for_review=sum(1 for e in events if e.needs_review),
                assumptions=factors.describe(),
            )

    @router.get("/dashboard/dishes", dependencies=staff, tags=["dashboard"])
    def dashboard_dishes(
        start: date | None = None, end: date | None = None, meal: str | None = None
    ) -> DishesOut:
        """Every dish seen from ``start`` to ``end`` inclusive (default: the last 7 days).

        Dish trends compare with the period of the same length just before ``start``.
        """
        last = end or today()
        first = start or last - timedelta(days=DEFAULT_DISHES_DAYS - 1)
        _check_range(first, last)
        stop = last + timedelta(days=1)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = store.events(first, stop, meal)
            previous = store.events(first - (stop - first), first, meal)
            return DishesOut(
                hall=hall,
                start=first,
                end=last,
                data_source=store.data_source(first, stop),
                rows=_rows(events, previous),
            )

    @router.get("/dashboard/dishes/{dish_name:path}", dependencies=staff, tags=["dashboard"])
    def dashboard_dish(
        dish_name: str,
        start: date | None = None,
        end: date | None = None,
        meal: str | None = None,
    ) -> DishDetailOut:
        """One dish from ``start`` to ``end`` inclusive (default: the last 28 days).

        404 if the dish has no matched regions in that window.
        """
        last = end or today()
        first = start or last - timedelta(days=DEFAULT_DETAIL_DAYS - 1)
        _check_range(first, last)
        stop = last + timedelta(days=1)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = store.events(first, stop, meal, dish=dish_name)
            if not events:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    f"no plates recorded for {dish_name!r} between {first} and {last}",
                )
            previous = store.events(first - (stop - first), first, meal, dish=dish_name)
            samples = [round(e.percent_left, 1) for e in events if e.percent_left is not None]
            stride = -(-len(samples) // MAX_SAMPLES) or 1
            return DishDetailOut(
                dish=_rows(events, previous)[0],
                data_source=store.data_source(first, stop),
                percent_left_samples=samples[::stride],
                weekly=[
                    TrendPoint(
                        week_start=p.week_start,
                        kg_wasted=round(p.kg_wasted, 3),
                        waste_rate_percent=_r(p.waste_rate_percent, 1),
                    )
                    for p in weekly_points(_observations(events))
                ],
            )

    @router.get("/dashboard/trends", dependencies=staff, tags=["dashboard"])
    def dashboard_trends(weeks: Annotated[int, Query(ge=1, le=52)] = 8) -> TrendsOut:
        """Weekly totals for the last ``weeks`` weeks, the current (partial) week included.

        Weeks start on Monday. Weeks without any plate are omitted, not shown as zero.
        """
        stop = today() + timedelta(days=1)
        first = week_start(today()) - timedelta(weeks=weeks - 1)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = store.events(first, stop)
            return TrendsOut(
                hall=hall,
                data_source=store.data_source(first, stop),
                weeks=[
                    TrendPoint(
                        week_start=p.week_start,
                        kg_wasted=round(p.kg_wasted, 2),
                        waste_rate_percent=_r(p.waste_rate_percent, 1),
                    )
                    for p in weekly_points(_observations(events))
                ],
            )

    @router.get("/report/weekly", tags=["report"])
    def report_weekly(
        week_start_day: Annotated[date | None, Query(alias="week_start")] = None,
    ) -> ReportOut:
        """Public, aggregated weekly report for USC Student Government.

        ``week_start`` may be any day; the report covers the Monday-to-Sunday week
        containing it. Default: the last complete week. Only per-dish totals are
        returned, never individual plates.
        """
        first = week_start(week_start_day or today() - timedelta(days=7))
        stop = first + timedelta(days=7)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = store.events(first, stop)
            previous = store.events(first - timedelta(days=7), first)
            source = store.data_source(first, stop)
            return ReportOut(
                hall=hall,
                week_start=first,
                data_source=source,
                headline=_headline(events, factors),
                top_dishes=_rows(events, previous)[:TOP_DISHES],
                methods_note=METHODS_NOTE + (SEEDED_NOTE if source == "seeded" else ""),
                assumptions=factors.describe(),
            )

    return router
