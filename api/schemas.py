"""The contract between the rig, the API and the dashboard.

Everything that crosses a process boundary is defined here and nowhere else.
The dashboard's TypeScript types are generated from ``/openapi.json``, so change
this file first, then tell the other lanes.

No schema here has a field that identifies a student, a card or a person.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.types import FoodCategory

DataSource = Literal["live", "seeded"]
"""``seeded`` marks demo data we generated. Every screen must say so when it is shown."""


class RegionIn(BaseModel):
    """One measured food region on a plate, as the rig reports it."""

    model_config = ConfigDict(extra="forbid")

    category: FoodCategory
    dish_name: str | None = Field(None, description="None means unknown (goes to review)")
    match_score: float | None = Field(None, ge=0.0, le=1.0)
    volume_ml: float = Field(ge=0.0)
    grams: float = Field(ge=0.0)
    grams_low: float = Field(ge=0.0)
    grams_high: float = Field(ge=0.0)
    percent_left: float | None = Field(
        None, ge=0.0, description="Volume left over the dish's reference volume; may exceed 100"
    )

    @model_validator(mode="after")
    def _range_brackets_estimate(self) -> Self:
        if not self.grams_low <= self.grams <= self.grams_high:
            raise ValueError("need grams_low <= grams <= grams_high")
        return self


class PlateIn(BaseModel):
    """Body of ``POST /plates``: one returned plate and its regions."""

    model_config = ConfigDict(extra="forbid")

    plate_id: str = Field(
        min_length=8, max_length=64, description="Random id; not derived from anything personal"
    )
    meal: str = Field(min_length=1)
    observed_at: datetime = Field(
        description="The server rounds this down to 15 minutes before storing"
    )
    regions: list[RegionIn]


class PlateAccepted(BaseModel):
    plate_id: str
    regions_stored: int
    regions_for_review: int


class PlateImageAccepted(PlateAccepted):
    """``POST /plates/image``: the stored counts plus what was measured, for the capture page."""

    regions: list[RegionIn]


class Headline(BaseModel):
    """The three numbers at the top of the Today screen."""

    kg_wasted: float
    waste_rate_percent: float | None = Field(
        None, description="Mean percent left across matched regions"
    )
    est_cost_usd: float | None = Field(
        None, description="From editable cost factors; an assumption, say so"
    )
    est_co2e_kg: float | None = None


class DishRow(BaseModel):
    dish_name: str
    station: str | None = None
    category: FoodCategory | None = None
    plates_seen: int
    total_kg: float
    mean_percent_left: float | None
    trend: Literal["up", "down", "flat", "unknown"] = "unknown"


class TodayOut(BaseModel):
    """``GET /dashboard/today``."""

    hall: str
    day: date
    meal: str | None
    data_source: DataSource
    headline: Headline
    top_dishes: list[DishRow] = Field(description="Up to 5, most kilograms wasted first")
    plates_seen: int
    regions_for_review: int
    assumptions: dict[str, str] = Field(
        default_factory=dict,
        description="Cost and CO2e factors behind the headline estimates, named and explained",
    )


class DishesOut(BaseModel):
    """``GET /dashboard/dishes``: the sortable table and the popularity scatter."""

    hall: str
    start: date
    end: date
    data_source: DataSource
    rows: list[DishRow]


class DishDetailOut(BaseModel):
    """``GET /dashboard/dishes/{dish_name}``."""

    dish: DishRow
    data_source: DataSource
    percent_left_samples: list[float] = Field(description="For the distribution chart")
    weekly: list[TrendPoint]


class TrendPoint(BaseModel):
    """One week of one series; ``kg_wasted`` and ``waste_rate_percent`` as in ``Headline``."""

    week_start: date
    kg_wasted: float
    waste_rate_percent: float | None


class TrendsOut(BaseModel):
    """``GET /dashboard/trends``."""

    hall: str
    data_source: DataSource
    weeks: list[TrendPoint]


class ReportOut(BaseModel):
    """``GET /report/weekly``: everything the USG page and its PDF show."""

    hall: str
    week_start: date
    data_source: DataSource
    headline: Headline
    top_dishes: list[DishRow]
    methods_note: str
    assumptions: dict[str, str] = Field(description="Cost and CO2e factors, named and explained")


DishDetailOut.model_rebuild()
