"""Cost and CO2e factors: editable assumptions, never measurements.

Both estimates are ``kilograms wasted x factor``. The defaults below are
placeholders chosen to be round and plausible for a demo; they are not USC
figures. Edit them here, or override them without touching code through the
environment variables ``LEFTOVER_COST_USD_PER_KG`` and
``LEFTOVER_CO2E_KG_PER_KG``. Every response that shows a cost or CO2e number
ships ``describe()`` alongside it where the schema allows.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

COST_ENV = "LEFTOVER_COST_USD_PER_KG"
CO2E_ENV = "LEFTOVER_CO2E_KG_PER_KG"


@dataclass(frozen=True, slots=True)
class CostFactors:
    cost_usd_per_kg: float = 4.00
    """Placeholder food cost per kilogram of plate waste, in US dollars."""
    co2e_kg_per_kg: float = 2.50
    """Placeholder kg CO2e per kilogram of plate waste, one blended factor for all food."""

    def __post_init__(self) -> None:
        if self.cost_usd_per_kg < 0 or self.co2e_kg_per_kg < 0:
            raise ValueError("cost and CO2e factors cannot be negative")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> CostFactors:
        env = os.environ if environ is None else environ
        default = cls()
        return cls(
            cost_usd_per_kg=float(env.get(COST_ENV, default.cost_usd_per_kg)),
            co2e_kg_per_kg=float(env.get(CO2E_ENV, default.co2e_kg_per_kg)),
        )

    def est_cost_usd(self, kg_wasted: float) -> float:
        return kg_wasted * self.cost_usd_per_kg

    def est_co2e_kg(self, kg_wasted: float) -> float:
        return kg_wasted * self.co2e_kg_per_kg

    def describe(self) -> dict[str, str]:
        """The ``assumptions`` map: each factor named, valued and explained."""
        return {
            "cost_usd_per_kg": (
                f"{self.cost_usd_per_kg:.2f} USD per kg of plate waste. Assumed placeholder, "
                f"not a USC Hospitality figure; set {COST_ENV} to their food cost per kg."
            ),
            "co2e_kg_per_kg": (
                f"{self.co2e_kg_per_kg:.2f} kg CO2e per kg of plate waste. Assumed placeholder, "
                f"one blended factor for all dishes, not measured; set {CO2E_ENV} to a "
                "sourced factor."
            ),
            "basis": (
                "Estimated cost and CO2e are kg wasted multiplied by these factors. The kg "
                "figure is itself an estimate (measured volume x a per-category density), so "
                "both numbers are indicative only."
            ),
        }
