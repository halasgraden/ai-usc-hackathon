from __future__ import annotations

from datetime import datetime

from analytics.aggregates import DishSummary, PlateObservation, flag_high_waste, summarize_dishes

NOW = datetime(2026, 11, 14, 18, 0)


def obs(dish: str | None, grams: float, pct: float | None) -> PlateObservation:
    return PlateObservation("university-village", "Dinner", NOW, dish, grams, pct)


def test_summaries_total_grams_per_dish_and_skip_unknowns() -> None:
    summaries = summarize_dishes(
        [obs("Rice Pilaf", 100, 50), obs("Rice Pilaf", 300, 100), obs(None, 80, None)]
    )
    assert summaries == [
        DishSummary("Rice Pilaf", plates_seen=2, total_kg=0.4, mean_percent_left=75.0)
    ]


def test_flagging_uses_the_threshold() -> None:
    summaries = [DishSummary("A", 3, 1.0, 60.0), DishSummary("B", 3, 1.0, 20.0)]
    assert flag_high_waste(summaries, threshold_percent=40.0) == ["A"]
