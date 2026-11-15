"""The live path behind POST /plates/image, with every model replaced by a fake."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from api.main import create_app
from api.models import Plate, PlateEvent
from api.pipeline import (
    PlatePipeline,
    Unmeasurable,
    build_pipeline,
    decode_image,
    merge_by_category,
)
from api.settings import Settings
from capture.scale import PlateEllipse
from core.types import DishMatch, FoodCategory, FoodRegion, ImageBGR, Mask
from depth.volume import PlateMeasurement
from portion.grams import load_densities

H = W = 64
DEVICE = {"Authorization": "Bearer device-secret"}


def _box(y0: int, y1: int, x0: int, x1: int) -> Mask:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


PLATE = _box(4, 60, 4, 60)
RICE = FoodCategory.RICE_GRAINS
FRIES = FoodCategory.POTATOES_FRIED_ROASTED


class FakeSegmenter:
    def __init__(self, regions: list[FoodRegion], plate: Mask | None = PLATE) -> None:
        self.regions, self.plate = regions, plate

    def segment(self, image: ImageBGR) -> tuple[list[FoodRegion], Mask | None]:
        return self.regions, self.plate


class FakeMatcher:
    def __init__(self, names: dict[FoodCategory, str | None]) -> None:
        self.names = names

    def match(self, image: ImageBGR, region: FoodRegion) -> DishMatch:
        name = self.names.get(region.category)
        return DishMatch(name, 0.7 if name else 0.0)


class Recorder:
    """A measurer that returns 10 ml per 100 px and remembers what it was given."""

    def __init__(self, result: str = "ok") -> None:
        self.result = result
        self.calls: list[list[Mask]] = []

    def __call__(
        self, image: ImageBGR, plate: Mask, foods: Sequence[Mask]
    ) -> PlateMeasurement | None:
        self.calls.append(list(foods))
        if self.result == "none":
            return None
        return PlateMeasurement(
            volumes_ml=tuple(float(m.sum()) / 10.0 for m in foods),
            cm_per_px=0.4,
            focal_px=50.0,
            plate_distance_cm=20.0,
            depth_scale=1.0,
            ellipse=PlateEllipse((32.0, 32.0), 56.0, 56.0, 0.0),
        )


def _pipeline(
    regions: list[FoodRegion],
    *,
    plate: Mask | None = PLATE,
    measure: Recorder | None = None,
    names: dict[FoodCategory, str | None] | None = None,
    meals: tuple[str, ...] = ("Dinner",),
    reference: float | None = None,
) -> PlatePipeline:
    matcher = FakeMatcher(names if names is not None else {RICE: "Rice Pilaf"})

    def matcher_for(meal: str) -> FakeMatcher:
        if meal not in meals:
            raise KeyError(meal)
        return matcher

    return PlatePipeline(
        segmenter=FakeSegmenter(regions, plate),
        measure=measure or Recorder(),
        matcher_for=matcher_for,
        reference_volume=lambda _dish: reference,
    )


IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


def test_merge_by_category_unions_masks_and_keeps_best_confidence() -> None:
    a, b = _box(10, 20, 10, 20), _box(15, 25, 15, 25)
    merged = merge_by_category(
        [FoodRegion(a, FRIES, 0.4), FoodRegion(_box(30, 40, 30, 40), RICE, 0.9),
         FoodRegion(b, FRIES, 0.6)]
    )  # fmt: skip
    assert [r.category for r in merged] == [FRIES, RICE]
    assert merged[0].confidence == 0.6
    assert merged[0].mask.sum() == (a | b).sum()  # overlap counted once
    assert merge_by_category([FoodRegion(np.zeros((H, W), bool), RICE, 0.9)]) == []


def test_run_measures_grams_and_matches_each_class_once() -> None:
    measure = Recorder()
    regions = [
        FoodRegion(_box(10, 20, 10, 20), RICE, 0.9),  # 100 px -> 10 ml
        FoodRegion(_box(30, 40, 30, 40), FRIES, 0.8),  # 100 px
        FoodRegion(_box(35, 45, 30, 40), FRIES, 0.5),  # overlaps the fries above
    ]
    out = _pipeline(regions, measure=measure, names={RICE: "Rice Pilaf", FRIES: None}).run(
        IMAGE, "Dinner"
    )
    assert len(measure.calls) == 1 and len(measure.calls[0]) == 2  # one mask per class
    rice, fries = out
    density = load_densities()[RICE]
    assert rice.dish_name == "Rice Pilaf" and rice.match_score == 0.7
    assert rice.volume_ml == 10.0
    assert rice.grams == pytest.approx(round(10.0 * density.value, 1))
    assert rice.grams_low <= rice.grams <= rice.grams_high
    assert rice.percent_left is None  # no reference portion yet: not invented
    assert fries.dish_name is None and fries.match_score is None  # unknown -> review
    assert fries.volume_ml == 15.0  # union of two overlapping boxes: 150 px


def test_percent_left_uses_the_reference_volume_when_there_is_one() -> None:
    (rice,) = _pipeline([FoodRegion(_box(10, 20, 10, 20), RICE, 0.9)], reference=40.0).run(
        IMAGE, "Dinner"
    )
    assert rice.percent_left == 25.0  # 10 ml left of a 40 ml reference


def test_clean_plate_is_an_empty_list_and_skips_measuring() -> None:
    measure = Recorder()
    assert _pipeline([], measure=measure).run(IMAGE, "Dinner") == []
    assert measure.calls == []


def test_unmeasurable_frames_raise() -> None:
    with pytest.raises(Unmeasurable, match="no plate"):
        _pipeline([], plate=None).run(IMAGE, "Dinner")
    with pytest.raises(Unmeasurable, match="scale"):
        _pipeline([FoodRegion(_box(10, 20, 10, 20), RICE, 0.9)], measure=Recorder("none")).run(
            IMAGE, "Dinner"
        )


def test_unknown_meal_fails_before_any_model_runs() -> None:
    measure = Recorder()
    with pytest.raises(KeyError):
        _pipeline([FoodRegion(_box(10, 20, 10, 20), RICE, 0.9)], measure=measure).run(
            IMAGE, "Midnight Snack"
        )
    assert measure.calls == []


def test_decode_image_round_trips_and_rejects_garbage() -> None:
    ok, buf = cv2.imencode(".png", np.full((5, 7, 3), 200, dtype=np.uint8))
    assert ok
    assert decode_image(bytes(buf)).shape == (5, 7, 3)
    with pytest.raises(ValueError, match="not a readable"):
        decode_image(b"not an image")


def test_build_pipeline_refuses_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_pipeline(tmp_path / "nope.onnx", hall="h", data_dir=tmp_path, today=date.today)


# --- the endpoint ---------------------------------------------------------------


def _png() -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((H, W, 3), dtype=np.uint8))
    assert ok
    return bytes(buf)


def _client(tmp_path: Path, pipeline: PlatePipeline) -> tuple[Any, TestClient]:
    settings = Settings(
        hall="test-hall",
        database_url=f"sqlite:///{tmp_path / 'p.db'}",
        data_dir=tmp_path,
        device_token="device-secret",
        staff_token="staff-secret",
    )
    app = create_app(settings, pipeline=lambda: pipeline)
    return app, TestClient(app)


def _post(client: TestClient, data: bytes | None = None, meal: str = "Dinner") -> Any:
    files = {"image": ("plate.png", data if data is not None else _png(), "image/png")}
    return client.post("/plates/image", headers=DEVICE, files=files, data={"meal": meal})


def test_photo_is_measured_stored_and_echoed(tmp_path: Path) -> None:
    regions = [
        FoodRegion(_box(10, 20, 10, 20), RICE, 0.9),
        FoodRegion(_box(30, 40, 30, 40), FRIES, 0.8),
    ]
    app, client = _client(tmp_path, _pipeline(regions, names={RICE: "Rice Pilaf"}))
    response = _post(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["regions_stored"] == 2 and body["regions_for_review"] == 1
    assert [r["dish_name"] for r in body["regions"]] == ["Rice Pilaf", None]
    with Session(app.state.db.engine) as session:
        assert len(session.exec(select(Plate)).all()) == 1
        events = session.exec(select(PlateEvent)).all()
    assert {e.dish_name for e in events} == {"Rice Pilaf", None}
    assert all(e.data_source == "live" for e in events)


def test_clean_plate_photo_is_stored_with_no_regions(tmp_path: Path) -> None:
    app, client = _client(tmp_path, _pipeline([]))
    body = _post(client).json()
    assert body["regions_stored"] == 0 and body["regions"] == []
    with Session(app.state.db.engine) as session:
        assert len(session.exec(select(Plate)).all()) == 1


def test_bad_photos_are_rejected_and_nothing_is_stored(tmp_path: Path) -> None:
    app, client = _client(tmp_path, _pipeline([], plate=None))
    assert _post(client, b"garbage").status_code == 422
    no_plate = _post(client)
    assert no_plate.status_code == 422 and "no plate" in no_plate.json()["detail"]
    wrong_meal = _post(client, meal="Midnight Snack")
    assert wrong_meal.status_code == 422 and "not on today's menu" in wrong_meal.json()["detail"]
    with Session(app.state.db.engine) as session:
        assert session.exec(select(Plate)).all() == []
