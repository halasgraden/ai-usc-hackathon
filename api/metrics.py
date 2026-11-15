"""Shared dashboard arithmetic. Bounds are conservative density ranges, never CIs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date, timedelta

from analytics.aggregates import (
    PlateObservation,
    mean_percent_left,
    summarize_dishes,
    trend_direction,
    week_start,
)
from analytics.assumptions import CostFactors
from api.dishes import CATALOG, station_of
from api.models import PlateEvent, ServingRecord
from api.schemas import DishRow, Headline, ServingComparison, TrendPoint
from core.types import FoodCategory


def observations(events: Sequence[PlateEvent]) -> list[PlateObservation]:
    return [
        PlateObservation(
            e.hall, e.meal, e.observed_at, e.dish_name, e.grams, e.percent_left, e.plate_id
        )
        for e in events
    ]


def comparison(
    events: Sequence[PlateEvent], servings: Sequence[ServingRecord]
) -> ServingComparison | None:
    # No partial-cohort denominator and no live/seeded mixing, even when names match.
    if not servings or any(e.data_source != "seeded" for e in events):
        return None
    if any(s.data_source != "seeded" or not s.capture_complete for s in servings):
        return None
    keys = {(s.hall, s.day, s.meal, s.dish_name) for s in servings}
    matched = [e for e in events if e.dish_name is not None]
    if any((e.hall, e.observed_at.date(), e.meal, e.dish_name) not in keys for e in matched):
        return None
    served = sum(s.served_kg for s in servings)
    if served <= 0:
        return None
    return ServingComparison(
        served_kg=round(served, 6),
        servings=sum(s.servings for s in servings),
        returned_percent=round(sum(e.grams for e in matched) / (served * 10), 4),
        returned_percent_low=round(sum(e.grams_low for e in matched) / (served * 10), 4),
        returned_percent_high=round(sum(e.grams_high for e in matched) / (served * 10), 4),
    )


def headline(
    events: Sequence[PlateEvent], servings: Sequence[ServingRecord], factors: CostFactors
) -> Headline:
    kg = sum(e.grams for e in events) / 1000
    index = mean_percent_left(observations(events))
    return Headline(
        kg_wasted=round(kg, 6),
        kg_low=round(sum(e.grams_low for e in events) / 1000, 6),
        kg_high=round(sum(e.grams_high for e in events) / 1000, 6),
        waste_rate_percent=None if index is None else round(index, 1),
        est_cost_usd=round(factors.est_cost_usd(kg), 2),
        est_co2e_kg=round(factors.est_co2e_kg(kg), 2),
        serving_comparison=comparison(events, servings),
    )


def rows(
    current: Sequence[PlateEvent],
    previous: Sequence[PlateEvent],
    servings: Sequence[ServingRecord] = (),
) -> list[DishRow]:
    before = {s.dish_name: s for s in summarize_dishes(observations(previous))}
    groups: dict[str, list[PlateEvent]] = {}
    for event in current:
        if event.dish_name is not None:
            groups.setdefault(event.dish_name, []).append(event)
    serving_groups: dict[str, list[ServingRecord]] = {}
    for serving in servings:
        serving_groups.setdefault(serving.dish_name, []).append(serving)
        groups.setdefault(serving.dish_name, [])
    result = []
    for name, events in groups.items():
        n = len({e.plate_id for e in events})
        kg = sum(e.grams for e in events) / 1000
        low, high = (
            sum(getattr(e, field) for e in events) / 1000 for field in ("grams_low", "grams_high")
        )
        index = mean_percent_left(observations(events))
        prior = before.get(name)
        category = Counter(e.category for e in events).most_common(1)
        result.append(
            DishRow(
                dish_name=name,
                station=station_of(name),
                category=FoodCategory(category[0][0])
                if category
                else CATALOG[name].category
                if name in CATALOG
                else None,
                plates_seen=n,
                total_kg=round(kg, 6),
                kg_low=round(low, 6),
                kg_high=round(high, 6),
                mean_percent_left=None if index is None else round(index, 1),
                mean_grams=round(kg * 1000 / n, 4) if n else None,
                mean_grams_low=round(low * 1000 / n, 4) if n else None,
                mean_grams_high=round(high * 1000 / n, 4) if n else None,
                previous_total_kg=round(prior.total_kg, 6) if prior else None,
                previous_mean_percent_left=round(prior.mean_percent_left, 1)
                if prior and prior.mean_percent_left is not None
                else None,
                trend=trend_direction(index, prior.mean_percent_left if prior else None),
                serving_comparison=comparison(events, serving_groups.get(name, [])),
            )
        )
    return sorted(result, key=lambda r: (-r.total_kg, r.dish_name))


def weekly(
    events: Sequence[PlateEvent],
    servings: Sequence[ServingRecord],
    start: date,
    end: date,
    today: date,
) -> list[TrendPoint]:
    groups: dict[date, list[PlateEvent]] = {}
    served_groups: dict[date, list[ServingRecord]] = {}
    for event in events:
        groups.setdefault(week_start(event.observed_at.date()), []).append(event)
    for record in servings:
        key = week_start(record.day)
        served_groups.setdefault(key, []).append(record)
        groups.setdefault(key, [])
    result = []
    for day, group in sorted(groups.items()):
        h = headline(group, served_groups.get(day, []), CostFactors())
        result.append(
            TrendPoint(
                week_start=day,
                kg_wasted=h.kg_wasted,
                kg_low=h.kg_low,
                kg_high=h.kg_high,
                serving_comparison=h.serving_comparison,
                waste_rate_percent=h.waste_rate_percent,
                partial=day < start
                or day + timedelta(days=6) > min(end, today - timedelta(days=1)),
            )
        )
    return result
