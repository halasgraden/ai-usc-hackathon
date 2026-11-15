from __future__ import annotations

import yaml

from core.types import SEG_CLASSES
from vision.train import DEFAULT_CONFIG


def test_class_list_is_plate_bowl_then_the_21_food_classes() -> None:
    assert SEG_CLASSES[:2] == ("plate", "bowl")
    assert len(SEG_CLASSES) == 23 and len(set(SEG_CLASSES)) == 23
    assert SEG_CLASSES.index("rice_grains") == 2


def test_training_config_is_committed_and_reproducible() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert {"model", "data", "imgsz", "epochs", "seed"} <= set(config)
