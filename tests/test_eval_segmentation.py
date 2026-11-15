from __future__ import annotations

from pathlib import Path

from core.types import SEG_CLASSES
from eval.segmentation import MIN_INSTANCES, ClassResult, count_instances, results_markdown


def test_count_instances_counts_masks_and_images(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("0 0 0 1 0 1 1\n0 0 0 1 0 1 1\n3 0 0 1 0 1 1\n")
    (tmp_path / "b.txt").write_text("3 0 0 1 0 1 1\n\n")
    instances, images = count_instances(tmp_path)
    assert len(instances) == len(SEG_CLASSES)
    assert (instances[0], images[0]) == (2, 1)
    assert (instances[3], images[3]) == (2, 2)
    assert sum(instances) == 4


def test_results_markdown_flags_thin_classes() -> None:
    rows = [
        ClassResult("plate", 19, 19, 0.9, 0.8, 0.95, 0.7),
        ClassResult("poultry", MIN_INSTANCES - 1, 2, 0.5, 0.5, 0.4, 0.2),
    ]
    md = results_markdown("v1", {"mask_ap50": 0.6, "mask_ap50_95": 0.4}, rows)
    assert "| plate | 19 | 19 | 0.900 | 0.800 | 0.950 | 0.700 |" in md
    assert "| poultry * |" in md
    assert "mask mAP50 0.600" in md
    assert rows[0].reliable and not rows[1].reliable
