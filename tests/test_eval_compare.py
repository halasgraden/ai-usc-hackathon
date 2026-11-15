from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.compare import compare_markdown, load


def _run(
    name: str, ap50: float, classes: list[tuple[str, float, bool]], imgsz: int | None = None
) -> dict[str, Any]:
    run: dict[str, Any] = {
        "name": name,
        "overall": {"mask_ap50": ap50, "mask_ap50_95": ap50 / 2},
        "classes": [
            {"name": n, "mask_ap50": v, "mask_ap50_95": v / 2, "reliable": ok}
            for n, v, ok in classes
        ],
    }
    if imgsz:
        run["imgsz"] = imgsz
    return run


def test_compare_lines_up_classes_and_marks_gaps() -> None:
    a = _run("a", 0.5, [("plate", 0.9, True), ("poultry", 0.4, False)])
    b = _run("b", 0.6, [("plate", 0.95, True), ("rice_grains", 0.7, True)], imgsz=1024)
    md = compare_markdown([a, b])
    assert "| | a | b (1024 px) |" in md
    assert "| Mask mAP50, all classes | 0.500 | 0.600 |" in md
    assert "| plate AP50 | 0.900 | 0.950 |" in md
    assert "| poultry * AP50 | 0.400 | - |" in md  # unreliable in run a, missing in b
    assert "| rice_grains AP50 | - | 0.700 |" in md


def test_load_reads_a_results_file(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps(_run("x", 0.1, [])))
    assert load("x", tmp_path)["name"] == "x"
