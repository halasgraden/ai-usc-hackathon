from __future__ import annotations

import numpy as np
import pytest

from augment.composite import (
    Cutout,
    build_composites,
    composite,
    extract_cutouts,
    photometric_jitter,
)
from labeling.autolabel import LabeledMask

H = W = 200


def _disc(cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return np.asarray((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, dtype=np.bool_)


def _plate_photo(food: bool = False) -> tuple[np.ndarray, list[LabeledMask]]:
    image = np.full((H, W, 3), 60, dtype=np.uint8)
    plate = _disc(100, 100, 80)
    image[plate] = 230
    labels = [LabeledMask("plate", 0.9, plate)]
    if food:
        patch = np.zeros((H, W), dtype=bool)
        patch[90:110, 70:130] = True
        image[patch] = (0, 200, 0)  # green food (BGR)
        labels.append(LabeledMask("cooked_vegetables", 0.8, patch))
    return image, labels


def _bank() -> list[Cutout]:
    image, labels = _plate_photo(food=True)
    return extract_cutouts(image, labels)


def test_extract_cutouts_skips_plate_and_records_source_plate_size() -> None:
    cuts = _bank()
    assert [c.class_name for c in cuts] == ["cooked_vegetables"]
    assert cuts[0].mask.shape == (20, 60) and cuts[0].mask.all()
    assert cuts[0].plate_px == pytest.approx(np.sqrt(_disc(100, 100, 80).sum()))


def test_extract_cutouts_ignores_tiny_masks() -> None:
    image, labels = _plate_photo()
    tiny = np.zeros((H, W), dtype=bool)
    tiny[5:8, 5:8] = True
    labels.append(LabeledMask("poultry", 0.9, tiny))
    assert extract_cutouts(image, labels) == []


def test_composite_pastes_food_inside_the_plate_with_matching_labels() -> None:
    base, base_labels = _plate_photo()
    out, labels = composite(base, base_labels, _bank(), np.random.default_rng(0), n_foods=(2, 2))
    assert out.shape == base.shape and not np.array_equal(out, base)
    classes = [m.class_name for m in labels]
    assert classes[0] == "plate" and set(classes[1:]) == {"cooked_vegetables"}
    plate = labels[0].mask
    for m in labels[1:]:
        assert (m.mask & plate).sum() >= 0.9 * m.mask.sum()
        assert out[m.mask].mean(axis=0)[1] > 150  # pasted pixels are the green food
    assert np.array_equal(base, _plate_photo()[0])  # the input was not modified


def test_pasted_foods_do_not_overlap_each_other_much() -> None:
    base, base_labels = _plate_photo()
    _, labels = composite(
        base, base_labels, _bank(), np.random.default_rng(3), n_foods=(3, 3), max_overlap=0.0
    )
    foods = [m.mask for m in labels[1:]]
    for i, a in enumerate(foods):
        for b in foods[i + 1 :]:
            assert not (a & b).any()


def test_composite_is_deterministic_for_a_seed() -> None:
    base, base_labels = _plate_photo()
    a, la = composite(base, base_labels, _bank(), np.random.default_rng(7))
    b, lb = composite(base, base_labels, _bank(), np.random.default_rng(7))
    assert np.array_equal(a, b) and [m.class_name for m in la] == [m.class_name for m in lb]


def test_composite_needs_a_plate() -> None:
    base, _ = _plate_photo()
    with pytest.raises(ValueError, match="plate"):
        composite(base, [], _bank(), np.random.default_rng(0))


def test_scale_follows_plate_size_ratio() -> None:
    # Same cutout, source plate half the size of the base plate -> pasted ~2x larger per side.
    cut = _bank()[0]
    small = Cutout(cut.class_name, cut.image, cut.mask, cut.plate_px / 2)
    base, base_labels = _plate_photo()
    _, labels = composite(
        base, base_labels, [small], np.random.default_rng(1), n_foods=(1, 1), jitter=0.0
    )
    ys, xs = np.nonzero(labels[1].mask)
    assert (xs.max() - xs.min() + 1) > 60 * 1.5 or (ys.max() - ys.min() + 1) > 60 * 1.5


def test_photometric_jitter_keeps_shape_and_never_swaps_the_dominant_colour() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[...] = (30, 180, 60)  # green-dominant (BGR)
    rng = np.random.default_rng(0)
    for _ in range(25):
        out = photometric_jitter(image, rng)
        assert out.shape == image.shape and out.dtype == np.uint8
        assert out.reshape(-1, 3).mean(axis=0).argmax() == 1


def test_build_composites_returns_n_labeled_images() -> None:
    base = _plate_photo()
    made = build_composites([base], _bank(), 4, seed=1)
    assert len(made) == 4
    assert all(labels[0].class_name == "plate" for _, labels in made)
