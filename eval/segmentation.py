"""Evaluate a segmentation checkpoint on the held-out plates and write the numbers to a file.

This script is the only source of segmentation numbers in the README: it writes
``eval/results/<name>.json`` and ``eval/results/<name>.md``, and those files are what get
quoted. Nobody types a number by hand.

    python -m eval.segmentation --weights best.pt --data /content/datasets/village/data.yaml \
        --name village-v1

Validation images are real held-out plate groups (whole plates the model never trained on).
Classes with fewer than ``MIN_INSTANCES`` validation instances are reported but flagged: a
score on two masks is noise, not a result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from core.types import SEG_CLASSES

RESULTS_DIR = Path(__file__).parent / "results"
MIN_INSTANCES = 5


@dataclass(frozen=True, slots=True)
class ClassResult:
    name: str
    instances: int
    images: int
    precision: float
    recall: float
    mask_ap50: float
    mask_ap50_95: float

    @property
    def reliable(self) -> bool:
        return self.instances >= MIN_INSTANCES


def count_instances(
    labels_dir: Path, n_classes: int = len(SEG_CLASSES)
) -> tuple[list[int], list[int]]:
    """Per-class (instance count, image count) over every label file in ``labels_dir``."""
    instances, images = [0] * n_classes, [0] * n_classes
    for path in sorted(labels_dir.glob("*.txt")):
        seen: set[int] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls = int(line.split()[0])
                instances[cls] += 1
                seen.add(cls)
        for cls in seen:
            images[cls] += 1
    return instances, images


def results_markdown(
    name: str, overall: dict[str, float], rows: list[ClassResult], imgsz: int | None = None
) -> str:
    """The table pasted into the README, rendered from the JSON numbers."""
    lines = [
        f"### Segmentation: {name}",
        "",
        f"Held-out plates{f' at {imgsz} px' if imgsz else ''}, "
        f"mask mAP50 {overall['mask_ap50']:.3f}, mask mAP50-95 {overall['mask_ap50_95']:.3f}.",
        "",
        "| Class | Val images | Val instances | Precision | Recall | Mask AP50 | Mask AP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        flag = "" if r.reliable else " *"
        lines.append(
            f"| {r.name}{flag} | {r.images} | {r.instances} | {r.precision:.3f} | "
            f"{r.recall:.3f} | {r.mask_ap50:.3f} | {r.mask_ap50_95:.3f} |"
        )
    lines += [
        "",
        f"\\* fewer than {MIN_INSTANCES} validation instances: too few to trust.",
        "Classes with no validation instances are not listed; they have no held-out score.",
    ]
    return "\n".join(lines) + "\n"


def _val_labels_dir(data_yaml: Path) -> Path:
    cfg: dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    return root / str(cfg["val"]).replace("images", "labels", 1)


def evaluate(
    weights: Path,
    data_yaml: Path,
    name: str,
    out_dir: Path = RESULTS_DIR,
    imgsz: int | None = None,
) -> Path:
    """Validate, write ``<name>.json`` and ``<name>.md`` to ``out_dir``; return the JSON path."""
    from ultralytics import YOLO  # heavy; imported only when evaluating

    model = YOLO(str(weights))
    # None: evaluate at the size the model was trained at (stored in the checkpoint).
    size = int(imgsz if imgsz is not None else model.overrides.get("imgsz", 640))
    metrics = model.val(data=str(data_yaml), split="val", imgsz=size, plots=False)
    instances, images = count_instances(_val_labels_dir(data_yaml))
    rows: list[ClassResult] = []
    for i, cls in enumerate(metrics.ap_class_index):
        p, r, ap50, ap = metrics.seg.class_result(i)
        rows.append(
            ClassResult(
                SEG_CLASSES[int(cls)],
                instances[int(cls)],
                images[int(cls)],
                float(p),
                float(r),
                float(ap50),
                float(ap),
            )
        )
    overall = {"mask_ap50": float(metrics.seg.map50), "mask_ap50_95": float(metrics.seg.map)}
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    payload = {
        "name": name,
        "weights": str(weights),
        "data": str(data_yaml),
        "imgsz": size,
        "overall": overall,
        "classes": [asdict(r) | {"reliable": r.reliable} for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{name}.md").write_text(
        results_markdown(name, overall, rows, size), encoding="utf-8"
    )
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    parser.add_argument("--name", required=True, help="results file name, e.g. village-v1")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--imgsz", type=int, default=None, help="default: the training size")
    args = parser.parse_args()
    path = evaluate(args.weights, args.data, args.name, args.out, args.imgsz)
    print(path.with_suffix(".md").read_text(encoding="utf-8"))
    print("written:", path, path.with_suffix(".md"))


if __name__ == "__main__":
    main()
