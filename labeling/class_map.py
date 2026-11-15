"""Label-name normalization and the Voxel51 -> FoodCategory mapping."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.types import FoodCategory

DEFAULT_CLASS_MAP = Path(__file__).parent / "class_map.yaml"


def normalize_label(label: str) -> str:
    """Canonical form of a dataset label.

    The real Voxel51 data contains the typo ``egg-spaetzle)``.

    >>> normalize_label(" Egg-Spaetzle) ")
    'egg-spaetzle'
    """
    return label.strip().casefold().replace(")", "").replace("(", "").strip()


def load_class_map(path: Path = DEFAULT_CLASS_MAP) -> dict[str, FoodCategory | None]:
    """Read the mapping file; keys are normalized labels.

    Raises:
        ValueError: a value is not a FoodCategory (or null).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["voxel51"]
    return {
        normalize_label(label): (FoodCategory(target) if target is not None else None)
        for label, target in data.items()
    }
