"""Shared vocabulary: food categories and the records passed between pipeline stages.

The 21 food categories are exactly the rows of the density table in the product
spec. They are the label space of the segmentation model, the keys of
``portion/densities.yaml`` and the targets of the menu categorizer, so they are
defined once, here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageBGR: TypeAlias = NDArray[np.uint8]
"""H x W x 3 image in OpenCV channel order."""
Mask: TypeAlias = NDArray[np.bool_]
"""H x W boolean mask, True where the region is."""
DepthMap: TypeAlias = NDArray[np.float32]
"""H x W distance from the camera; units depend on the stage (see ``depth``)."""


class FoodCategory(StrEnum):
    """Coarse food classes (one per row of the spec's density table)."""

    RICE_GRAINS = "rice_grains"
    PASTA_PLAIN = "pasta_plain"
    PASTA_SAUCY = "pasta_saucy"
    SOUP_STEW = "soup_stew"
    MASHED_PUREE = "mashed_puree"
    BEANS_LENTILS = "beans_lentils"
    POULTRY = "poultry"
    RED_MEAT = "red_meat"
    FISH_SEAFOOD = "fish_seafood"
    EGGS = "eggs"
    TOFU_PLANT_PROTEIN = "tofu_plant_protein"
    RAW_VEGETABLES = "raw_vegetables"
    COOKED_VEGETABLES = "cooked_vegetables"
    LEAFY_SALAD = "leafy_salad"
    POTATOES_FRIED_ROASTED = "potatoes_fried_roasted"
    BREAD = "bread"
    PIZZA_BURGER_SANDWICH = "pizza_burger_sandwich"
    BAKED_DESSERT = "baked_dessert"
    ICE_CREAM = "ice_cream"
    YOGURT_PUDDING = "yogurt_pudding"
    CUT_FRUIT = "cut_fruit"


SEG_CLASSES: tuple[str, ...] = ("plate", "bowl", *(c.value for c in FoodCategory))
"""Class list of the segmentation model, in training-label index order."""


@dataclass(frozen=True, slots=True)
class ScaleCalibration:
    """Pixel-to-centimetre scale recovered from the ArUco marker in a frame."""

    cm_per_px: float


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """One sharp, face-free frame of a plate, ready for the vision stages."""

    image: ImageBGR
    captured_at: datetime
    sharpness: float
    scale: ScaleCalibration | None


@dataclass(frozen=True, slots=True)
class FoodRegion:
    """One segmented food instance on a plate."""

    mask: Mask
    category: FoodCategory
    confidence: float


@dataclass(frozen=True, slots=True)
class DishMatch:
    """Result of matching a region against today's menu. ``dish_name`` None = unknown."""

    dish_name: str | None
    score: float


@dataclass(frozen=True, slots=True)
class PortionEstimate:
    """Volume and the gram range it implies for one food region."""

    volume_ml: float
    grams: float
    grams_low: float
    grams_high: float
