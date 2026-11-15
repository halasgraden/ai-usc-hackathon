from __future__ import annotations

import numpy as np
import pytest

from core.types import SEG_CLASSES, FoodCategory
from labeling.autolabel import (
    LabeledMask,
    box_iou,
    build_prompt,
    filter_labels,
    load_pilot_items,
    missing_items,
    overlap_fraction,
    parse_photo_name,
    resolve_label,
    split_groups,
    split_items,
    to_yolo_seg_lines,
)

ITEMS = load_pilot_items()


def _rect(x0: int, y0: int, x1: int, y1: int, size: int = 100) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_pilot_items_cover_the_twelve_chosen_dishes_with_valid_categories() -> None:
    assert len(ITEMS) == 11  # fries covers both "Fries" and "Assorted French Fries"
    assert ITEMS["fries"].category is FoodCategory.POTATOES_FRIED_ROASTED
    assert ITEMS["short-ribs"].category is FoodCategory.RED_MEAT
    assert all(it.phrase.strip() for it in ITEMS.values())


def test_parse_photo_name() -> None:
    assert parse_photo_name("/x/Fries_P2_scraps_03.JPG") == ("fries", "fries_p2")
    assert parse_photo_name("empty.jpg") == ("empty", "empty")


def test_build_prompt_for_food_and_for_empty_plate() -> None:
    prompt, lookup = build_prompt("fries", ITEMS)
    assert prompt == "plate . bowl . french fries ."
    assert lookup["french fries"] == "potatoes_fried_roasted" and lookup["plate"] == "plate"
    empty_prompt, empty_lookup = build_prompt("empty", ITEMS)
    assert empty_prompt == "plate . bowl ." and "french fries" not in empty_lookup
    with pytest.raises(KeyError):
        build_prompt("lasagna", ITEMS)


def test_resolve_label_handles_partial_and_unknown_phrases() -> None:
    _, lookup = build_prompt("fries", ITEMS)
    assert resolve_label("french fries", lookup) == "potatoes_fried_roasted"
    assert resolve_label("fries plate", lookup) == "plate"  # unknown word + known phrase
    assert resolve_label("fork", lookup) is None


def test_box_iou_and_overlap_fraction() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert overlap_fraction(_rect(0, 0, 10, 10), _rect(0, 0, 5, 10)) == pytest.approx(0.5)
    assert overlap_fraction(np.zeros((5, 5), dtype=bool), _rect(0, 0, 5, 5, 5)) == 0.0


def test_filter_keeps_best_plate_and_food_on_it() -> None:
    plate = _rect(10, 10, 90, 90)
    labels = [
        LabeledMask("plate", 0.9, plate),
        LabeledMask("plate", 0.4, _rect(0, 0, 100, 100)),  # worse plate: dropped
        LabeledMask("potatoes_fried_roasted", 0.5, _rect(30, 30, 60, 60)),
    ]
    kept = filter_labels(labels)
    assert [m.class_name for m in kept] == ["plate", "potatoes_fried_roasted"]
    assert kept[0].score == 0.9


def test_filter_drops_food_off_plate_whole_plate_and_slivers() -> None:
    plate = _rect(10, 10, 90, 90)
    labels = [
        LabeledMask("plate", 0.9, plate),
        LabeledMask("poultry", 0.6, _rect(0, 0, 8, 8)),  # napkin beside the plate
        LabeledMask("poultry", 0.6, plate.copy()),  # DINO boxed the whole plate
        LabeledMask("poultry", 0.6, _rect(50, 50, 52, 52)),  # sliver (4 px)
    ]
    assert [m.class_name for m in filter_labels(labels)] == ["plate"]


def test_filter_prefers_the_big_plate_over_a_high_scoring_sliver() -> None:
    labels = [
        LabeledMask("plate", 0.64, _rect(0, 0, 90, 8)),  # strip at the frame edge
        LabeledMask("plate", 0.52, _rect(10, 20, 90, 95)),  # the real plate
        LabeledMask("rice_grains", 0.7, _rect(30, 40, 60, 70)),
    ]
    kept = filter_labels(labels)
    assert [m.class_name for m in kept] == ["plate", "rice_grains"]
    assert kept[0].score == 0.52


def test_filter_keeps_food_when_the_plate_mask_has_a_hole_under_it() -> None:
    ring = _rect(10, 10, 90, 90)
    ring[30:70, 30:70] = False  # SAM segmented the ceramic only, not the food on it
    labels = [
        LabeledMask("plate", 0.8, ring),
        LabeledMask("poultry", 0.8, _rect(35, 35, 65, 65)),
    ]
    assert [m.class_name for m in filter_labels(labels)] == ["plate", "poultry"]


def test_filter_drops_a_bowl_on_the_plate_and_keeps_the_food_it_covered() -> None:
    labels = [
        LabeledMask("plate", 0.8, _rect(10, 10, 90, 90)),
        LabeledMask("bowl", 0.5, _rect(30, 30, 60, 60)),  # the rice pile, called a bowl
        LabeledMask("rice_grains", 0.7, _rect(31, 31, 59, 59)),
    ]
    assert [m.class_name for m in filter_labels(labels)] == ["plate", "rice_grains"]


def test_filter_keeps_a_real_bowl_beside_the_plate() -> None:
    labels = [
        LabeledMask("plate", 0.8, _rect(0, 0, 50, 50)),
        LabeledMask("bowl", 0.6, _rect(55, 55, 95, 95)),
    ]
    assert [m.class_name for m in filter_labels(labels)] == ["plate", "bowl"]


def test_filter_drops_duplicate_masks_of_the_same_class_only() -> None:
    labels = [
        LabeledMask("plate", 0.8, _rect(5, 5, 95, 95)),
        LabeledMask("cooked_vegetables", 0.4, _rect(20, 20, 50, 50)),
        LabeledMask("cooked_vegetables", 0.5, _rect(21, 21, 50, 50)),  # same spears again
        LabeledMask("poultry", 0.6, _rect(22, 22, 48, 48)),  # other class: kept
        LabeledMask("cooked_vegetables", 0.3, _rect(60, 60, 80, 80)),  # another spear: kept
    ]
    kept = filter_labels(labels)
    veg = [m.score for m in kept if m.class_name == "cooked_vegetables"]
    assert veg == [0.5, 0.3]
    assert "poultry" in [m.class_name for m in kept]


def test_missing_items_names_foods_with_no_mask() -> None:
    items = load_pilot_items()
    labels = [
        LabeledMask("plate", 0.8, _rect(0, 0, 9, 9)),
        LabeledMask("rice_grains", 0.7, _rect(1, 1, 5, 5)),
    ]
    assert missing_items("chicken+rice-pilaf", labels, items) == ["chicken"]
    assert missing_items("rice-pilaf", labels, items) == []
    assert missing_items("empty", labels, items) == []
    veg = [LabeledMask("cooked_vegetables", 0.5, _rect(1, 1, 5, 5))]
    assert missing_items("asparagus+carrots", veg, items) == []


def test_filter_drops_food_when_no_plate_was_found() -> None:
    assert filter_labels([LabeledMask("poultry", 0.9, _rect(30, 30, 60, 60))]) == []


def test_yolo_lines_use_the_model_class_index() -> None:
    lines = to_yolo_seg_lines([LabeledMask("plate", 0.9, _rect(10, 10, 90, 90))])
    assert len(lines) == 1 and lines[0].startswith(f"{SEG_CLASSES.index('plate')} ")
    assert to_yolo_seg_lines([LabeledMask("bowl", 0.9, np.zeros((50, 50), dtype=bool))]) == []


def test_split_groups_is_deterministic_and_holds_out_whole_groups() -> None:
    groups = [f"fries_p{i}" for i in range(1, 6)] + [f"rice-pilaf_p{i}" for i in range(1, 4)]
    groups += ["crepe_p1"]
    val = split_groups(groups * 4)  # photos repeat groups
    assert val == split_groups(groups)
    assert val <= set(groups)
    assert any(g.startswith("fries") for g in val) and any(g.startswith("rice-pilaf") for g in val)
    assert "crepe_p1" not in val  # a single group stays in training


def test_mixed_plate_prompt_asks_for_every_food() -> None:
    assert split_items("asparagus+chicken") == ["asparagus", "chicken"]
    assert split_items("empty") == ["empty"]
    prompt, lookup = build_prompt("asparagus+chicken", ITEMS)
    assert prompt == "plate . bowl . asparagus . chicken ."
    assert lookup["asparagus"] == "cooked_vegetables" and lookup["chicken"] == "poultry"
    with pytest.raises(KeyError):
        build_prompt("fries+lasagna", ITEMS)


def test_mixed_plate_name_parses_and_groups_by_its_first_item() -> None:
    assert parse_photo_name("asparagus+chicken_p1_half_04.heic") == (
        "asparagus+chicken",
        "asparagus+chicken_p1",
    )
    val = split_groups(["asparagus+chicken_p1", "asparagus+chicken_p2", "asparagus+chicken_p3"])
    assert len(val) == 1
