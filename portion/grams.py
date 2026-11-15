"""Convert a region's volume to grams, with the range its density implies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from core.types import FoodCategory, PortionEstimate

DEFAULT_DENSITIES_PATH = Path(__file__).parent / "densities.yaml"


@dataclass(frozen=True, slots=True)
class Density:
    """Bulk density of one food class in g/ml, with its plausible range."""

    value: float
    low: float
    high: float
    basis: str
    source: str


@lru_cache(maxsize=4)
def load_densities(path: Path = DEFAULT_DENSITIES_PATH) -> dict[FoodCategory, Density]:
    """Load the density table.

    Raises:
        ValueError: a food category is missing, unknown, or has a bad value.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    spread: dict[str, float] = data["relative_uncertainty"]
    table: dict[FoodCategory, Density] = {}
    for name, row in data["densities"].items():
        value, basis = float(row["density"]), str(row["basis"])
        if value <= 0 or basis not in spread:
            raise ValueError(f"bad density entry for {name!r}")
        table[FoodCategory(name)] = Density(
            value=value,
            low=value * (1 - spread[basis]),
            high=value * (1 + spread[basis]),
            basis=basis,
            source=str(row.get("source", "")),
        )
    missing = set(FoodCategory) - set(table)
    if missing:
        raise ValueError(f"densities missing for: {sorted(m.value for m in missing)}")
    return table


def estimate_grams(
    volume_ml: float, category: FoodCategory, path: Path = DEFAULT_DENSITIES_PATH
) -> PortionEstimate:
    """Grams = volume x the class's bulk density, with the low/high range.

    Raises:
        ValueError: negative volume.
    """
    if volume_ml < 0:
        raise ValueError("volume cannot be negative")
    density = load_densities(path)[category]
    return PortionEstimate(
        volume_ml=volume_ml,
        grams=volume_ml * density.value,
        grams_low=volume_ml * density.low,
        grams_high=volume_ml * density.high,
    )


def percent_left(volume_left_ml: float, served_volume_ml: float) -> float:
    """Volume left as a percentage of the dish's reference served volume.

    Density cancels out, so this does not depend on the density table. It is
    not capped at 100: a plate holding more than the reference portion is real
    information (self-served portions vary), not an error.

    Raises:
        ValueError: non-positive served volume or negative volume left.
    """
    if served_volume_ml <= 0:
        raise ValueError("served volume must be positive")
    if volume_left_ml < 0:
        raise ValueError("volume left cannot be negative")
    return 100.0 * volume_left_ml / served_volume_ml
