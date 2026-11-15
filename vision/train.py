"""Train the segmentation model from a YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "yolo26n_seg.yaml"


def load_train_config(config_path: Path = DEFAULT_CONFIG, **overrides: Any) -> dict[str, Any]:
    """Read the training config and apply keyword overrides (e.g. ``data=``, ``epochs=``).

    Raises:
        ValueError: the config has no ``model`` entry.
    """
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(overrides)
    if "model" not in config:
        raise ValueError(f"{config_path} has no 'model' entry")
    return config


def train(config_path: Path = DEFAULT_CONFIG, **overrides: Any) -> Path:
    """Train YOLO26n-seg with the hyperparameters in ``config_path``.

    Returns:
        Path to the best checkpoint (``<project>/<name>/weights/best.pt``).
    """
    from ultralytics import YOLO  # heavy; imported only when training

    config = load_train_config(config_path, **overrides)
    model = YOLO(config.pop("model"))
    model.train(**config)
    return Path(model.trainer.best)
