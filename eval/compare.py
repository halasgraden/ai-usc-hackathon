"""Side-by-side table of segmentation runs, rendered from their ``eval/results/*.json`` files.

    python -m eval.compare village-v1 village-v2 > eval/results/compare-v1-v2.md

Only numbers present in the result files are printed; a class a run did not score shows "-".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.segmentation import RESULTS_DIR


def load(name: str, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """One run's results JSON (written by ``eval.segmentation``)."""
    data: dict[str, Any] = json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))
    return data


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def compare_markdown(runs: list[dict[str, Any]]) -> str:
    """Overall and per-class mask AP50 / AP50-95 for each run, classes in first-seen order."""
    names = [r["name"] for r in runs]
    sizes = [f"{r['name']} ({r['imgsz']} px)" if "imgsz" in r else r["name"] for r in runs]
    header = "| | " + " | ".join(sizes) + " |"
    rule = "|---|" + "---:|" * len(runs)
    lines = [f"### Segmentation runs compared: {', '.join(names)}", "", header, rule]
    for key, label in (
        ("mask_ap50", "Mask mAP50, all classes"),
        ("mask_ap50_95", "Mask mAP50-95, all classes"),
    ):
        lines.append(f"| {label} | " + " | ".join(_fmt(r["overall"][key]) for r in runs) + " |")
    classes: list[str] = []
    for r in runs:
        classes += [c["name"] for c in r["classes"] if c["name"] not in classes]
    for cls in classes:
        rows = [{c["name"]: c for c in r["classes"]}.get(cls) for r in runs]
        flag = "" if all(row is None or row["reliable"] for row in rows) else " *"
        for key, label in (("mask_ap50", "AP50"), ("mask_ap50_95", "AP50-95")):
            values = " | ".join(_fmt(row[key] if row else None) for row in rows)
            lines.append(f"| {cls}{flag} {label} | {values} |")
    lines += ["", "\\* fewer than 5 validation instances in at least one run: too few to trust."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="result names, e.g. village-v1 village-v2")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    print(compare_markdown([load(n, args.results) for n in args.names]), end="")


if __name__ == "__main__":
    main()
