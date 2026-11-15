from __future__ import annotations

from datetime import date, datetime

import pytest

from analytics.aggregates import (
    DishSummary,
    PlateObservation,
    WeekPoint,
    flag_high_waste,
    mean_percent_left,
    summarize_dishes,
    total_kg,
    trend_direction,
    week_start,
    weekly_points,
)
from analytics.assumptions import CostFactors


def obs(
    dish: str | None,
    grams: float,
    pct: float | None,
    at: datetime = datetime(2026, 11, 14, 18, 0),
    plate: str | None = None,
) -> PlateObservation:
    return PlateObservation("university-village", "Dinner", at, dish, grams, pct, plate)


def test_summaries_sort_by_kg_then_name() -> None:
    summaries = summarize_dishes(
        [obs("B", 100, 10), obs("A", 100, 30), obs("C", 500, None), obs("A", 50, None)]
    )
    assert [s.dish_name for s in summaries] == ["C", "A", "B"]
    assert summaries[0].mean_percent_left is None  # no percent values at all
    assert summaries[1] == DishSummary("A", 2, 0.15, 30.0)


def test_two_regions_of_one_dish_on_one_plate_count_as_one_plate() -> None:
    summaries = summarize_dishes([obs("A", 10, 5, plate="p1"), obs("A", 20, 15, plate="p1")])
    assert [(s.dish_name, s.plates_seen, s.mean_percent_left) for s in summaries] == [
        ("A", 1, 10.0)
    ]
    assert summaries[0].total_kg == pytest.approx(0.03)


def test_empty_input() -> None:
    assert summarize_dishes([]) == []
    assert total_kg([]) == 0.0
    assert mean_percent_left([]) is None
    assert weekly_points([]) == []


def test_total_counts_unknowns_but_waste_rate_does_not() -> None:
    items = [obs("A", 100, 20), obs(None, 300, 90)]
    assert total_kg(items) == pytest.approx(0.4)
    assert mean_percent_left(items) == 20.0


def test_flags_skip_dishes_without_percent_and_keep_order() -> None:
    summaries = [DishSummary("A", 1, 1.0, None), DishSummary("B", 1, 1.0, 41.0),
                 DishSummary("C", 1, 1.0, 40.0), DishSummary("D", 1, 0.5, 90.0)]  # fmt: skip
    assert flag_high_waste(summaries, threshold_percent=40.0) == ["B", "D"]


def test_week_start_is_monday() -> None:
    assert week_start(date(2026, 11, 15)) == date(2026, 11, 9)  # Sunday
    assert week_start(date(2026, 11, 9)) == date(2026, 11, 9)  # Monday


def test_weekly_points_skip_empty_weeks() -> None:
    points = weekly_points(
        [
            obs("A", 100, 10, at=datetime(2026, 11, 15, 12)),
            obs("A", 300, 30, at=datetime(2026, 11, 9, 8)),
            obs(None, 200, None, at=datetime(2026, 10, 27, 8)),
        ]
    )
    assert points == [
        WeekPoint(date(2026, 10, 26), 0.2, None),
        WeekPoint(date(2026, 11, 9), 0.4, 20.0),
    ]


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [(30.0, 20.0, "up"), (20.0, 30.0, "down"), (21.0, 20.0, "flat"), (22.0, 20.0, "up"),
     (None, 20.0, "unknown"), (20.0, None, "unknown")],
)  # fmt: skip
def test_trend_direction(current: float | None, previous: float | None, expected: str) -> None:
    assert trend_direction(current, previous) == expected


def test_cost_factors_multiply_and_explain_themselves() -> None:
    factors = CostFactors(cost_usd_per_kg=5.0, co2e_kg_per_kg=2.0)
    assert factors.est_cost_usd(3.0) == 15.0
    assert factors.est_co2e_kg(3.0) == 6.0
    described = factors.describe()
    assert set(described) == {"cost_usd_per_kg", "co2e_kg_per_kg", "basis"}
    assert "5.00" in described["cost_usd_per_kg"]
    assert "placeholder" in described["co2e_kg_per_kg"].lower()


def test_cost_factors_read_the_environment() -> None:
    env = {"LEFTOVER_COST_USD_PER_KG": "6.5", "LEFTOVER_CO2E_KG_PER_KG": "1.25"}
    assert CostFactors.from_env(env) == CostFactors(6.5, 1.25)
    assert CostFactors.from_env({}) == CostFactors()
    with pytest.raises(ValueError):
        CostFactors(cost_usd_per_kg=-1.0)
