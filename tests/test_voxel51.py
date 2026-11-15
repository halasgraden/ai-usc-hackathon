from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from core.types import SEG_CLASSES
from labeling.class_map import load_class_map
from labeling.voxel51 import convert, render_summary


def make_sample(tmp_path: Path, name: str, split: str, labels: list[str]) -> SimpleNamespace:
    path = tmp_path / f"{name}.jpg"
    cv2.imwrite(str(path), np.full((100, 200, 3), 128, dtype=np.uint8))
    detections = [
        SimpleNamespace(
            label=label,
            bounding_box=(0.1 + 0.3 * i, 0.2, 0.25, 0.5),
            mask=np.ones((20, 20), dtype=bool),
        )
        for i, label in enumerate(labels)
    ]
    return SimpleNamespace(
        filepath=str(path), split=split, yoloe_segmentation=SimpleNamespace(detections=detections)
    )


def test_convert_writes_a_trainable_dataset(tmp_path: Path) -> None:
    samples = [
        make_sample(tmp_path, "a", "train", ["mashed-potatoes", "carrot", "bread-dumplings"]),
        make_sample(tmp_path, "b", "test", ["egg-spaetzle)", "never-seen-before"]),
    ]
    out = tmp_path / "ds"
    stats = convert(samples, out, load_class_map())

    assert dict(stats.images) == {"train": 1, "val": 1}
    label_files = sorted((out / "labels").rglob("*.txt"))
    assert len(label_files) == 2 and len(list((out / "images").rglob("*.jpg"))) == 2

    train_lines = (out / "labels" / "train").glob("*.txt")
    first = next(iter(train_lines)).read_text().splitlines()
    assert len(first) == 2  # the dumpling is dropped by the map
    assert {int(line.split()[0]) for line in first} == {
        SEG_CLASSES.index("mashed_puree"), SEG_CLASSES.index("cooked_vegetables")
    }  # fmt: skip
    coords = [float(v) for v in first[0].split()[1:]]
    assert len(coords) % 2 == 0 and all(0.0 <= v <= 1.0 for v in coords)

    assert stats.dropped_by_map["bread-dumplings"] == 1
    assert stats.unmapped_labels["never-seen-before"] == 1
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"][0] == "plate" and len(data["names"]) == len(SEG_CLASSES)
    assert json.loads((out / "stats.json").read_text())["images"] == {"train": 1, "val": 1}


def test_summary_names_the_classes_with_no_examples(tmp_path: Path) -> None:
    stats = convert(
        [make_sample(tmp_path, "a", "train", ["carrot"])], tmp_path / "ds", load_class_map()
    )
    assert "rice_grains" in stats.classes_without_examples()
    assert "cooked_vegetables" not in stats.classes_without_examples()
    assert "NO training examples for 20 of 21 classes" in render_summary(stats)


def test_unreadable_images_are_counted_not_fatal(tmp_path: Path) -> None:
    broken = SimpleNamespace(
        filepath=str(tmp_path / "missing.jpg"), split="train", yoloe_segmentation=None
    )
    stats = convert([broken], tmp_path / "ds", load_class_map())
    assert stats.unreadable_images == 1 and not stats.images


def test_limit_stops_early(tmp_path: Path) -> None:
    samples = [make_sample(tmp_path, f"s{i}", "train", ["carrot"]) for i in range(3)]
    assert sum(convert(samples, tmp_path / "ds", load_class_map(), limit=2).images.values()) == 2
