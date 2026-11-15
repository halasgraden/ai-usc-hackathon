from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from augment.dataset import COMPOSITE_PREFIX, add_composites, polygon_mask, read_yolo_labels
from core.types import SEG_CLASSES
from labeling.autolabel import LabeledMask, to_yolo_seg_lines

H = W = 200


def _disc(cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return np.asarray((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, dtype=np.bool_)


def _write(root: Path, split: str, name: str, food: bool) -> None:
    image = np.full((H, W, 3), 60, dtype=np.uint8)
    plate = _disc(100, 100, 80)
    image[plate] = 230
    labels = [LabeledMask("plate", 1.0, plate)]
    if food:
        patch = np.zeros((H, W), dtype=bool)
        patch[90:120, 70:130] = True
        image[patch] = (0, 200, 0)
        labels.append(LabeledMask("cooked_vegetables", 1.0, patch))
    for d in ("images", "labels"):
        (root / d / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), image)
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(to_yolo_seg_lines(labels)))


def test_polygon_mask_fills_the_square() -> None:
    m = polygon_mask([0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75], 100, 100)
    assert m[50, 50] and not m[10, 10]
    assert m.sum() == pytest.approx(51 * 51, rel=0.05)


def test_read_yolo_labels_round_trips_written_labels() -> None:
    plate = _disc(100, 100, 80)
    text = "\n".join(to_yolo_seg_lines([LabeledMask("plate", 0.7, plate)]))
    (back,) = read_yolo_labels(text, H, W)
    assert back.class_name == "plate"
    iou = (back.mask & plate).sum() / (back.mask | plate).sum()
    assert iou > 0.95


def test_read_yolo_labels_rejects_bad_lines() -> None:
    with pytest.raises(ValueError, match="SEG_CLASSES"):
        read_yolo_labels(f"{len(SEG_CLASSES)} 0 0 1 0 1 1", H, W)
    with pytest.raises(ValueError, match="3 x,y pairs"):
        read_yolo_labels("0 0.1 0.1 0.2 0.2", H, W)
    assert read_yolo_labels("\n\n", H, W) == []


def test_add_composites_writes_train_only_and_is_rerunnable(tmp_path: Path) -> None:
    _write(tmp_path, "train", "veg_p1_half_01", food=True)
    _write(tmp_path, "train", "empty_p2_clean_01", food=False)
    _write(tmp_path, "val", "veg_p9_half_01", food=True)
    assert add_composites(tmp_path, 5) == 5
    comps = sorted((tmp_path / "images" / "train").glob(f"{COMPOSITE_PREFIX}*.jpg"))
    assert len(comps) == 5
    assert not list((tmp_path / "images" / "val").glob(f"{COMPOSITE_PREFIX}*"))
    label = (tmp_path / "labels" / "train" / f"{comps[0].stem}.txt").read_text()
    classes = {SEG_CLASSES[int(line.split()[0])] for line in label.splitlines()}
    assert "plate" in classes and "cooked_vegetables" in classes
    assert add_composites(tmp_path, 3) == 3  # rerun replaces, does not pile up
    assert len(list((tmp_path / "images" / "train").glob(f"{COMPOSITE_PREFIX}*"))) == 3


def test_add_composites_needs_food_in_training(tmp_path: Path) -> None:
    _write(tmp_path, "train", "empty_p1_clean_01", food=False)
    assert add_composites(tmp_path, 4) == 0
