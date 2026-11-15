"""Add copy-paste composites to the training split of a YOLO segmentation dataset.

Reads ``images/train`` and ``labels/train`` of a dataset written by the autolabel notebook,
cuts every food instance out of the training photos, pastes them onto training plates and
writes the results back into the training split as ``comp_XXXX.jpg``. The validation split
is never read, so composites cannot leak validation plates into training.

    python -m augment.dataset --data /content/datasets/village --n 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from augment.composite import build_composites, extract_cutouts
from core.types import SEG_CLASSES, ImageBGR, Mask
from labeling.autolabel import PLATE, LabeledMask, to_yolo_seg_lines

COMPOSITE_PREFIX = "comp_"


def polygon_mask(coords: list[float], height: int, width: int) -> Mask:
    """Filled mask of one normalized YOLO polygon (``x1 y1 x2 y2 ...``)."""
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, 2) * (width, height)
    out = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(out, [np.round(pts).astype(np.int32)], 1)
    return out.astype(bool)


def read_yolo_labels(text: str, height: int, width: int) -> list[LabeledMask]:
    """Parse a YOLO segmentation label file into masks (score 1.0: these are labels).

    Raises ValueError on a class index outside ``SEG_CLASSES`` or a polygon with fewer than
    three points.
    """
    out: list[LabeledMask] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        if not 0 <= cls < len(SEG_CLASSES):
            raise ValueError(f"class index {cls} is not in SEG_CLASSES")
        coords = [float(v) for v in parts[1:]]
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"polygon needs at least 3 x,y pairs: {line[:60]!r}")
        out.append(LabeledMask(SEG_CLASSES[cls], 1.0, polygon_mask(coords, height, width)))
    return out


def load_split(root: Path, split: str) -> list[tuple[str, ImageBGR, list[LabeledMask]]]:
    """Every real (non-composite) photo of ``split`` with its labels."""
    out: list[tuple[str, ImageBGR, list[LabeledMask]]] = []
    for img_path in sorted((root / "images" / split).glob("*.jpg")):
        if img_path.name.startswith(COMPOSITE_PREFIX):
            continue
        raw = cv2.imread(str(img_path))
        if raw is None:
            raise ValueError(f"cannot read {img_path}")
        image: ImageBGR = np.asarray(raw, dtype=np.uint8)
        label_path = root / "labels" / split / (img_path.stem + ".txt")
        text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        out.append((img_path.name, image, read_yolo_labels(text, *image.shape[:2])))
    return out


def add_composites(root: Path, n: int, seed: int = 0) -> int:
    """Write ``n`` composites into ``root``'s training split; returns how many were written.

    Old composites are removed first, so rerunning gives the same dataset for the same seed.
    """
    img_dir, lbl_dir = root / "images" / "train", root / "labels" / "train"
    for old in list(img_dir.glob(f"{COMPOSITE_PREFIX}*")) + list(
        lbl_dir.glob(f"{COMPOSITE_PREFIX}*")
    ):
        old.unlink()
    train = load_split(root, "train")
    bank = [c for _, image, labels in train for c in extract_cutouts(image, labels)]
    bases = [
        (image, labels) for _, image, labels in train if any(m.class_name == PLATE for m in labels)
    ]
    if not bank or not bases:
        return 0
    written = 0
    for i, (image, labels) in enumerate(build_composites(bases, bank, n, seed=seed)):
        name = f"{COMPOSITE_PREFIX}{i:04d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / f"{name}.txt").write_text("\n".join(to_yolo_seg_lines(labels)), encoding="utf-8")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, required=True, help="dataset root (has images/, labels/)"
    )
    parser.add_argument("--n", type=int, default=200, help="composites to write")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(add_composites(args.data, args.n, args.seed), "composites written to the training split")


if __name__ == "__main__":
    main()
