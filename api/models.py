"""Database tables. A plate event stores a plate, never a person."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class PlateEvent(SQLModel, table=True):
    """One food region measured on one returned plate.

    There is deliberately no student, card or device-owner field. ``observed_at``
    is rounded down to 15 minutes before it is stored.
    """

    id: int | None = Field(default=None, primary_key=True)
    plate_id: str = Field(index=True, description="Random id shared by regions of one plate")
    hall: str = Field(index=True)
    meal: str
    observed_at: datetime = Field(index=True)
    dish_name: str | None = Field(default=None, index=True, description="None = unknown")
    category: str
    volume_ml: float
    grams: float
    grams_low: float
    grams_high: float
    percent_left: float | None = None
    match_score: float | None = None
    needs_review: bool = False


def round_to_quarter_hour(moment: datetime) -> datetime:
    """Round a timestamp down to the 15-minute mark, as the privacy design requires."""
    return moment.replace(minute=moment.minute - moment.minute % 15, second=0, microsecond=0)
