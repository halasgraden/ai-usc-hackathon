"""Convert the Voxel51 food-waste dataset to an Ultralytics segmentation dataset.

Runs in Colab (it needs ``fiftyone`` and the dataset download)::

    python -m labeling.voxel51 --out /content/datasets/leftover

Facts about the source data this code is written against (read from the dataset
itself, not assumed): 375 images; masks live in ``yoloe_segmentation`` as
FiftyOne ``Detections`` whose masks are cropped to their boxes; the masks were
produced automatically by YOLO-E, not drawn by hand; labels are 20 German-canteen
ingredient names; there is no plate or bowl class.

The summary printed at the end lists which of our classes received no examples.
Read it before trusting a model trained on this data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import yaml

from core.types import SEG_CLASSES, FoodCategory
from labeling.class_map import DEFAULT_CLASS_MAP, load_class_map, normalize_label
from labeling.masks import mask_to_polygon, place_box_mask, yolo_seg_line

HUB_ID = "Voxel51/food-waste-dataset"


@dataclass(slots=True)
class ConversionStats:
    """What a conversion produced. Written to ``stats.json`` for the data card."""

    images: Counter[str] = field(default_factory=Counter)
    instances: dict[str, Counter[str]] = field(
        default_factory=lambda: {"train": Counter(), "val": Counter()}
    )
    unmapped_labels: Counter[str] = field(default_factory=Counter)
    dropped_by_map: Counter[str] = field(default_factory=Counter)
    unusable_masks: int = 0
    unreadable_images: int = 0

    def classes_without_examples(self) -> list[str]:
        """Food classes with no training instance at all."""
        return [c.value for c in FoodCategory if self.instances["train"][c.value] == 0]

    def to_json(self) -> dict[str, Any]:
        return {
            "images": dict(self.images),
            "instances": {split: dict(counts) for split, counts in self.instances.items()},
            "unmapped_labels": dict(self.unmapped_labels),
            "dropped_by_map": dict(self.dropped_by_map),
            "unusable_masks": self.unusable_masks,
            "unreadable_images": self.unreadable_images,
            "food_classes_without_training_examples": self.classes_without_examples(),
        }


def convert(
    samples: Iterable[Any],
    out_dir: Path,
    mapping: dict[str, FoodCategory | None],
    *,
    limit: int | None = None,
    min_area_px: float = 64.0,
) -> ConversionStats:
    """Write ``images/{train,val}``, ``labels/{train,val}``, ``data.yaml`` and ``stats.json``.

    The dataset's own ``train`` split becomes ``train`` and its ``test`` split
    becomes ``val``. Each sample needs ``filepath``, ``split`` and
    ``yoloe_segmentation.detections`` (each with ``label``, ``bounding_box``, ``mask``).
    """
    stats = ConversionStats()
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        image = cv2.imread(str(sample.filepath))
        if image is None:
            stats.unreadable_images += 1
            continue
        height, width = image.shape[:2]
        split = "val" if sample.split == "test" else "train"
        stem = f"{index:05d}_{Path(sample.filepath).stem}"

        lines: list[str] = []
        detections = sample.yoloe_segmentation.detections if sample.yoloe_segmentation else []
        for det in detections:
            label = normalize_label(det.label)
            if label not in mapping:
                stats.unmapped_labels[label] += 1
                continue
            category = mapping[label]
            if category is None:
                stats.dropped_by_map[label] += 1
                continue
            if det.mask is None:
                stats.unusable_masks += 1
                continue
            full = place_box_mask(det.mask, tuple(det.bounding_box), width, height)
            polygon = mask_to_polygon(full, min_area_px=min_area_px)
            if polygon is None:
                stats.unusable_masks += 1
                continue
            lines.append(yolo_seg_line(SEG_CLASSES.index(category.value), polygon))
            stats.instances[split][category.value] += 1

        shutil.copy2(sample.filepath, out_dir / "images" / split / f"{stem}.jpg")
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats.images[split] += 1

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(SEG_CLASSES)),
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats.to_json(), indent=1), encoding="utf-8")
    return stats


def render_summary(stats: ConversionStats) -> str:
    lines = [f"images: {dict(stats.images)}", "instances per class (train / val):"]
    for category in FoodCategory:
        train, val = (
            stats.instances["train"][category.value],
            stats.instances["val"][category.value],
        )
        if train or val:
            lines.append(f"  {category.value:<24}{train:>5} /{val:>4}")
    missing = stats.classes_without_examples()
    lines.append(f"NO training examples for {len(missing)} of {len(FoodCategory)} classes:")
    lines.append("  " + ", ".join(missing))
    if stats.unmapped_labels:
        lines.append(f"labels missing from class_map.yaml: {dict(stats.unmapped_labels)}")
    lines.append(
        f"dropped by map: {dict(stats.dropped_by_map)}; unusable masks: {stats.unusable_masks}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from fiftyone.utils.huggingface import load_from_hub  # heavy; only needed here

    dataset = load_from_hub(HUB_ID)
    stats = convert(dataset, args.out, load_class_map(args.class_map), limit=args.limit)
    print(render_summary(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
