from __future__ import annotations

import numpy as np
import pytest

from core.types import FoodCategory
from labeling.class_map import load_class_map, normalize_label
from labeling.masks import mask_to_polygon, place_box_mask, yolo_seg_line


def test_normalize_label_handles_the_real_typo() -> None:
    assert normalize_label("egg-spaetzle)") == "egg-spaetzle"
    assert normalize_label("  Brown-Sauce ") == "brown-sauce"


def test_class_map_covers_every_label_in_the_real_dataset() -> None:
    real = [
        "dressing-portion",
        "coleslaw",
        "iceberg-lettuce",
        "vegetable-based-cream",
        "mashed-potatoes",
        "egg-spaetzle)",
        "light-sauce-or-white-sauce",
        "bread-dumplings",
        "brown-sauce",
        "lentil-stew",
        "green-beans",
        "schnitzel-or-cutlet",
        "gravy",
        "savoy-cabbage",
        "zucchini-or-courgette",
        "cauliflower",
        "chicken-strips",
        "carrot",
        "peas",
        "diced-potatoes",
    ]
    mapping = load_class_map()
    assert len(real) == 20
    assert all(normalize_label(label) in mapping for label in real)
    assert mapping["bread-dumplings"] is None
    assert mapping["egg-spaetzle"] is FoodCategory.PASTA_PLAIN
    assert {c for c in mapping.values() if c} == {
        FoodCategory.MASHED_PUREE, FoodCategory.PASTA_PLAIN, FoodCategory.COOKED_VEGETABLES,
        FoodCategory.SOUP_STEW, FoodCategory.RED_MEAT, FoodCategory.POULTRY,
        FoodCategory.POTATOES_FRIED_ROASTED, FoodCategory.RAW_VEGETABLES, FoodCategory.LEAFY_SALAD,
    }  # fmt: skip


def test_box_mask_lands_where_the_box_says() -> None:
    box_mask = np.ones((10, 20), dtype=bool)  # a crop twice as wide as tall
    full = place_box_mask(box_mask, (0.25, 0.5, 0.5, 0.25), width=80, height=40)
    ys, xs = np.nonzero(full)
    assert (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) == (20, 59, 20, 29)
    assert full.sum() == 40 * 10


def test_box_mask_is_clipped_to_the_image() -> None:
    full = place_box_mask(np.ones((4, 4), dtype=bool), (0.9, 0.9, 0.5, 0.5), width=10, height=10)
    assert full.shape == (10, 10) and full[9, 9] and full.sum() == 1


def test_degenerate_boxes_and_bad_input() -> None:
    assert not place_box_mask(np.ones((4, 4), dtype=bool), (0.5, 0.5, 0.0, 0.0), 10, 10).any()
    with pytest.raises(ValueError):
        place_box_mask(np.ones((4,), dtype=bool), (0, 0, 1, 1), 10, 10)
    with pytest.raises(ValueError):
        place_box_mask(np.ones((4, 4), dtype=bool), (0, 0, 1, 1), 0, 10)


def test_polygon_of_a_rectangle() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 40:120] = True
    polygon = mask_to_polygon(mask)
    assert polygon is not None and len(polygon) == 4
    xs, ys = zip(*polygon, strict=True)
    assert min(xs) == pytest.approx(40 / 200) and max(xs) == pytest.approx(119 / 200)
    assert min(ys) == pytest.approx(20 / 100) and max(ys) == pytest.approx(59 / 100)


def test_polygon_rejects_empty_and_tiny_masks() -> None:
    assert mask_to_polygon(np.zeros((50, 50), dtype=bool)) is None
    tiny = np.zeros((50, 50), dtype=bool)
    tiny[5:8, 5:8] = True
    assert mask_to_polygon(tiny, min_area_px=64) is None


def test_polygon_keeps_only_the_largest_blob() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[5:15, 5:15] = True
    mask[40:90, 40:90] = True
    polygon = mask_to_polygon(mask)
    assert polygon is not None
    assert min(x for x, _ in polygon) >= 0.39


def test_yolo_line_format() -> None:
    assert yolo_seg_line(7, [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]) == (
        "7 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000"
    )
