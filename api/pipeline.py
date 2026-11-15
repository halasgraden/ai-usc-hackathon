"""Photo in, measured regions out: the live path behind ``POST /plates/image``.

    segment (YOLO26n-seg, ONNX)  ->  merge masks per class  ->  depth + plate scale -> ml
    ->  grams (bulk density)     ->  match to today's menu (SigLIP)  ->  ``RegionIn`` list

Every stage is injected, so the orchestration is unit-tested with fakes; ``build_pipeline``
wires the real models from settings.

Masks of one class on a plate are merged before measuring. The model often returns a pile of
fries as several overlapping instances, and measuring each separately would count the overlap
twice. The trade-off: two different dishes of the same class on one plate (two chicken dishes)
are measured and matched as one region.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from api.schemas import RegionIn
from core.types import DishMatch, FoodCategory, FoodRegion, ImageBGR, Mask
from depth.volume import PlateMeasurement
from portion.grams import estimate_grams, percent_left

MIN_REGION_ML = 2.0
"""Regions measuring less than this are dropped before matching and storing. On real photos
small false detections measure ~0 ml; kept, they would log a 0 g leftover for a dish that was
never on the plate (a fries plate reported "Portobello Pot Roast", 0.0 ml)."""


class Unmeasurable(Exception):
    """The frame cannot be measured honestly (no plate, plate clipped, no scale)."""


class SegmenterLike(Protocol):
    def segment(self, image: ImageBGR) -> tuple[list[FoodRegion], Mask | None]: ...


class MatcherLike(Protocol):
    def match(self, image: ImageBGR, region: FoodRegion) -> DishMatch: ...


Measurer = Callable[[ImageBGR, Mask, Sequence[Mask]], PlateMeasurement | None]
"""(image, plate mask, food masks) -> volumes, or None when the frame must not be measured."""

MatcherFor = Callable[[str], MatcherLike]
"""Meal name -> a matcher over today's menu for that meal. Raises KeyError if not served."""

ReferenceVolume = Callable[[str], float | None]
"""Dish name -> its reference served volume in ml, or None when it has none yet."""


def merge_by_category(regions: Sequence[FoodRegion]) -> list[FoodRegion]:
    """One region per food class: masks OR-ed, confidence = the best instance's.

    Order follows each class's first appearance, so results are deterministic.
    """
    merged: dict[FoodCategory, FoodRegion] = {}
    for r in regions:
        prev = merged.get(r.category)
        if prev is None:
            merged[r.category] = FoodRegion(r.mask.copy(), r.category, r.confidence)
        else:
            merged[r.category] = FoodRegion(
                prev.mask | r.mask, r.category, max(prev.confidence, r.confidence)
            )
    return [m for m in merged.values() if m.mask.any()]


@dataclass
class PlatePipeline:
    segmenter: SegmenterLike
    measure: Measurer
    matcher_for: MatcherFor
    reference_volume: ReferenceVolume = field(default=lambda _dish: None)

    def run(self, image: ImageBGR, meal: str) -> list[RegionIn]:
        """Measured regions for one plate photo (an empty list is a clean plate).

        Raises:
            Unmeasurable: no plate in view, or the plate cannot set the scale.
            KeyError: ``meal`` is not on today's menu.
        """
        matcher = self.matcher_for(meal)  # fail on a bad meal before the expensive steps
        found, plate = self.segmenter.segment(image)
        if plate is None:
            raise Unmeasurable("no plate found in the photo")
        regions = merge_by_category(found)
        if not regions:
            return []
        measurement = self.measure(image, plate, [r.mask for r in regions])
        if measurement is None:
            raise Unmeasurable(
                "plate cannot set the scale (partly out of frame, or not a round plate)"
            )
        out: list[RegionIn] = []
        for region, volume in zip(regions, measurement.volumes_ml, strict=True):
            volume = max(0.0, float(volume))
            if volume < MIN_REGION_ML:
                continue
            portion = estimate_grams(volume, region.category)
            match = matcher.match(image, region)
            reference = self.reference_volume(match.dish_name) if match.dish_name else None
            out.append(
                RegionIn(
                    category=region.category,
                    dish_name=match.dish_name,
                    match_score=min(1.0, max(0.0, match.score)) if match.dish_name else None,
                    volume_ml=round(portion.volume_ml, 1),
                    grams=round(portion.grams, 1),
                    grams_low=round(portion.grams_low, 1),
                    grams_high=round(portion.grams_high, 1),
                    percent_left=(round(percent_left(volume, reference), 1) if reference else None),
                )
            )
        return out


def decode_image(data: bytes) -> ImageBGR:
    """JPEG/PNG bytes -> BGR image. Raises ValueError if the bytes are not an image."""
    import cv2

    raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError("upload is not a readable JPEG or PNG image")
    return np.asarray(raw, dtype=np.uint8)


def build_pipeline(
    seg_model: Path,
    *,
    hall: str,
    data_dir: Path,
    today: Callable[[], date],
    focal_px: float | None = None,
) -> PlatePipeline:
    """Wire the real models. Heavy imports happen here, not at app start.

    Raises:
        FileNotFoundError: the segmentation model file is missing.
    """
    if not seg_model.exists():
        raise FileNotFoundError(f"segmentation model not found: {seg_model}")
    from depth.estimator import DepthEstimator
    from depth.volume import measure_food_volumes
    from menu import load_daily_menu
    from menu.matcher import MenuMatcher, SiglipEmbedder
    from vision.infer import Segmenter

    estimator = DepthEstimator()
    embedder = SiglipEmbedder()

    def measure(image: ImageBGR, plate: Mask, foods: Sequence[Mask]) -> PlateMeasurement | None:
        return measure_food_volumes(image, plate, foods, estimator=estimator, focal_px=focal_px)

    @lru_cache(maxsize=8)
    def matcher(day: date, meal: str) -> MatcherLike:
        menu = load_daily_menu(day, hall=hall, data_dir=data_dir)
        return MenuMatcher(menu, embedder, meal=meal)

    return PlatePipeline(
        segmenter=Segmenter(seg_model),
        measure=measure,
        matcher_for=lambda meal: matcher(today(), meal),
    )
