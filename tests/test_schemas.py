from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from api.schemas import PlateIn, TodayOut


def region(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "category": "rice_grains", "dish_name": "Rice Pilaf", "match_score": 0.8,
        "volume_ml": 100.0, "grams": 73.0, "grams_low": 62.0, "grams_high": 84.0, "percent_left": 40.0,
    }  # fmt: skip
    return base | changes


def plate(*regions: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {"plate_id": "a1b2c3d4e5f6", "meal": "Dinner", "observed_at": datetime(2026, 11, 14, 18, 5).isoformat(),
            "regions": list(regions)} | changes  # fmt: skip


def test_a_valid_plate_parses() -> None:
    parsed = PlateIn.model_validate(plate(region(), region(dish_name=None, match_score=None)))
    assert parsed.regions[1].dish_name is None


@pytest.mark.parametrize(
    "bad",
    [
        region(grams_low=80.0),  # low above the estimate
        region(category="sushi"),
        region(volume_ml=-1.0),
        region(percent_left=-5.0),
        region(match_score=1.5),
        region(student_id="123"),  # extra fields are refused, so identifiers cannot slip in
    ],
)
def test_bad_regions_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PlateIn.model_validate(plate(bad))


def test_percent_left_may_exceed_100() -> None:
    assert (
        PlateIn.model_validate(plate(region(percent_left=140.0))).regions[0].percent_left == 140.0
    )


def test_plate_level_extras_are_refused() -> None:
    with pytest.raises(ValidationError):
        PlateIn.model_validate(plate(region(), uscard="x"))


def test_responses_must_declare_whether_data_is_seeded() -> None:
    with pytest.raises(ValidationError):
        TodayOut.model_validate({"hall": "h", "day": "2026-11-14", "meal": None, "headline": {"kg_wasted": 0.0},
                                 "top_dishes": [], "plates_seen": 0, "regions_for_review": 0})  # fmt: skip
