"""Read endpoints for the staff dashboard and the public weekly report."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, col, func, select

from analytics.aggregates import (
    week_start,
)
from analytics.assumptions import CostFactors
from api import metrics
from api.dishes import station_of
from api.models import Database, Plate, PlateEvent, ServingRecord
from api.schemas import (
    DataSource,
    DishDetailOut,
    DishesOut,
    DishWeeks,
    ReportOut,
    TodayOut,
    TrendsOut,
)

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

    def servings(self, start: date, end: date, meal: str | None = None) -> list[ServingRecord]:
        query = select(ServingRecord).where(
            ServingRecord.hall == self.hall,
            ServingRecord.day >= start,
            ServingRecord.day < end,
        )
        if meal:
            query = query.where(func.lower(ServingRecord.meal) == meal.strip().lower())
        return list(self.session.exec(query).all())

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
                headline=metrics.headline(events, store.servings(d, end, meal), factors),
                top_dishes=metrics.rows(events, previous, store.servings(d, end, meal))[
                    :TOP_DISHES
                ],
                plates_seen=store.plate_count(d, end, meal),
                regions_for_review=sum(1 for e in events if e.needs_review),
                assumptions=factors.describe(),
            )

    def filter_events(
        events: list[PlateEvent], station: str | None, search: str | None
    ) -> list[PlateEvent]:
        return [
            e
            for e in events
            if (not station or station_of(e.dish_name or "") == station)
            and (not search or search.casefold() in (e.dish_name or "").casefold())
        ]

    def filter_servings(
        records: list[ServingRecord], station: str | None, search: str | None
    ) -> list[ServingRecord]:
        return [
            r
            for r in records
            if (not station or station_of(r.dish_name) == station)
            and (not search or search.casefold() in r.dish_name.casefold())
        ]

    @router.get("/dashboard/dishes", dependencies=staff, tags=["dashboard"])
    def dashboard_dishes(
        start: date | None = None,
        end: date | None = None,
        meal: str | None = None,
        station: str | None = None,
        search: str | None = None,
    ) -> DishesOut:
        last = end or today()
        first = start or last - timedelta(days=DEFAULT_DISHES_DAYS - 1)
        _check_range(first, last)
        stop = last + timedelta(days=1)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            all_events = store.events(first, stop)
            events = filter_events(store.events(first, stop, meal), station, search)
            previous = filter_events(
                store.events(first - (stop - first), first, meal), station, search
            )
            servings = filter_servings(store.servings(first, stop, meal), station, search)
            return DishesOut(
                hall=hall,
                start=first,
                end=last,
                data_source=store.data_source(first, stop),
                rows=metrics.rows(events, previous, servings),
                headline=metrics.headline(events, servings, factors),
                plates_seen=len({e.plate_id for e in events})
                if station or search
                else store.plate_count(first, stop, meal),
                unmatched_kg=round(sum(e.grams for e in events if e.dish_name is None) / 1000, 6),
                assumptions=factors.describe(),
                stations=sorted({s for e in all_events if (s := station_of(e.dish_name or ""))}),
                meals=sorted({e.meal for e in all_events}),
            )

    @router.get("/dashboard/dishes/{dish_name:path}", dependencies=staff, tags=["dashboard"])
    def dashboard_dish(
        dish_name: str, start: date | None = None, end: date | None = None, meal: str | None = None
    ) -> DishDetailOut:
        last = end or today()
        first = start or last - timedelta(days=DEFAULT_DETAIL_DAYS - 1)
        _check_range(first, last)
        stop = last + timedelta(days=1)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = store.events(first, stop, meal, dish=dish_name)
            servings = [r for r in store.servings(first, stop, meal) if r.dish_name == dish_name]
            if not events and not servings:
                raise HTTPException(404, "No observations for this dish in the selected period.")
            previous = store.events(first - (stop - first), first, meal, dish=dish_name)
            samples = [round(e.percent_left, 1) for e in events if e.percent_left is not None]
            stride = -(-len(samples) // MAX_SAMPLES) or 1
            return DishDetailOut(
                dish=metrics.rows(events, previous, servings)[0],
                data_source=store.data_source(first, stop),
                percent_left_samples=samples[::stride],
                weekly=metrics.weekly(events, servings, first, last, today()),
            )

    @router.get("/dashboard/trends", dependencies=staff, tags=["dashboard"])
    def dashboard_trends(
        weeks: Annotated[int, Query(ge=1, le=52)] = 8,
        start: date | None = None,
        end: date | None = None,
        meal: str | None = None,
        station: str | None = None,
        search: str | None = None,
    ) -> TrendsOut:
        last = end or today()
        first = start or week_start(last) - timedelta(weeks=weeks - 1)
        _check_range(first, last)
        stop = last + timedelta(days=1)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = filter_events(store.events(first, stop, meal), station, search)
            servings = filter_servings(store.servings(first, stop, meal), station, search)
            names = sorted(
                {e.dish_name for e in events if e.dish_name} | {r.dish_name for r in servings}
            )
            return TrendsOut(
                hall=hall,
                data_source=store.data_source(first, stop),
                start=first,
                end=last,
                weeks=metrics.weekly(events, servings, first, last, today()),
                dishes=[
                    DishWeeks(
                        dish_name=name,
                        station=station_of(name),
                        weeks=metrics.weekly(
                            [e for e in events if e.dish_name == name],
                            [r for r in servings if r.dish_name == name],
                            first,
                            last,
                            today(),
                        ),
                    )
                    for name in names
                ],
            )

    @router.get("/report/weekly", tags=["report"])
    def report_weekly(
        week_start_day: Annotated[date | None, Query(alias="week_start")] = None,
        meal: str | None = None,
        station: str | None = None,
        search: str | None = None,
    ) -> ReportOut:
        first = week_start(week_start_day or today() - timedelta(days=7))
        stop = first + timedelta(days=7)
        with Session(db.engine) as session:
            store = _Store(session, hall)
            events = filter_events(store.events(first, stop, meal), station, search)
            previous = filter_events(
                store.events(first - timedelta(days=7), first, meal), station, search
            )
            servings = filter_servings(store.servings(first, stop, meal), station, search)
            source = store.data_source(first, stop)
            dish_rows = metrics.rows(events, previous, servings)
            return ReportOut(
                hall=hall,
                week_start=first,
                week_end=stop - timedelta(days=1),
                partial=stop > today(),
                data_source=source,
                headline=metrics.headline(events, servings, factors),
                top_dishes=dish_rows[:TOP_DISHES],
                rows=dish_rows,
                plates_seen=len({e.plate_id for e in events})
                if station or search
                else store.plate_count(first, stop, meal),
                unmatched_kg=round(sum(e.grams for e in events if e.dish_name is None) / 1000, 6),
                methods_note=METHODS_NOTE + (SEEDED_NOTE if source == "seeded" else ""),
                assumptions=factors.describe(),
            )

    return router
