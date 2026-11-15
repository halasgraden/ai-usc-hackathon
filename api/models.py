"""Database tables. A plate event stores a plate, never a person."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, SQLModel, create_engine


class Plate(SQLModel, table=True):
    """One returned plate. Kept even when it has no food regions (a clean plate).

    No student, card, device or face field. ``observed_at`` is local hall time,
    rounded down to 15 minutes.
    """

    id: int | None = Field(default=None, primary_key=True)
    plate_id: str = Field(unique=True, index=True, description="Random id from the rig")
    hall: str = Field(index=True)
    meal: str
    observed_at: datetime = Field(sa_type=DateTime(timezone=False), index=True)
    data_source: str = Field(default="live", index=True, description='"live" or "seeded"')


class PlateEvent(SQLModel, table=True):
    """One food region measured on one returned plate.

    There is deliberately no student, card or device-owner field. ``observed_at``
    is rounded down to 15 minutes before it is stored.
    """

    id: int | None = Field(default=None, primary_key=True)
    plate_id: str = Field(index=True, description="Random id shared by regions of one plate")
    hall: str = Field(index=True)
    meal: str
    observed_at: datetime = Field(sa_type=DateTime(timezone=False), index=True)
    dish_name: str | None = Field(default=None, index=True, description="None = unknown")
    category: str
    volume_ml: float
    grams: float
    grams_low: float
    grams_high: float
    percent_left: float | None = None
    match_score: float | None = None
    needs_review: bool = False
    data_source: str = Field(default="live", index=True, description='"live" or "seeded"')


def round_to_quarter_hour(moment: datetime) -> datetime:
    """Round a timestamp down to the 15-minute mark, as the privacy design requires."""
    return moment.replace(minute=moment.minute - moment.minute % 15, second=0, microsecond=0)


def make_engine(database_url: str) -> Engine:
    """Create the engine and the tables. ``sqlite://`` is one shared in-memory DB (tests)."""
    kwargs: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if database_url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, **kwargs)
    SQLModel.metadata.create_all(engine)
    return engine


class Database:
    """The engine, created (with its tables) on first use, not at import time.

    Importing ``api.main`` therefore never creates a database file.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._engine: Engine | None = None
        self._lock = threading.Lock()

    @property
    def engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = make_engine(self.url)
            return self._engine
