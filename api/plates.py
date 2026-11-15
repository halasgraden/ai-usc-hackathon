"""Plate ingestion: ``POST /plates`` (measured regions) and ``POST /plates/image`` (a photo)."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, func, select

from api.models import Database, Plate, PlateEvent, round_to_quarter_hour
from api.pipeline import PlatePipeline, Unmeasurable, decode_image
from api.schemas import PlateAccepted, PlateImageAccepted, PlateIn
from menu.importer import MenuUnavailableError


def to_local_naive(moment: datetime, timezone: str) -> datetime:
    """Express a timestamp in the hall's local time without tzinfo (how it is stored).

    Naive input is taken to already be hall-local time.
    """
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(ZoneInfo(timezone)).replace(tzinfo=None)


def _accepted(session: Session, plate_id: str) -> PlateAccepted:
    rows = session.exec(
        select(PlateEvent.needs_review, func.count())
        .where(PlateEvent.plate_id == plate_id)
        .group_by(col(PlateEvent.needs_review))
    ).all()
    counts = {bool(review): int(n) for review, n in rows}
    return PlateAccepted(
        plate_id=plate_id,
        regions_stored=counts.get(False, 0) + counts.get(True, 0),
        regions_for_review=counts.get(True, 0),
    )


def add_plate(
    session: Session,
    plate: PlateIn,
    *,
    hall: str,
    timezone: str,
    data_source: Literal["live", "seeded"] = "live",
) -> None:
    """Add one plate and its regions to ``session`` (the caller commits).

    The timestamp is converted to hall-local time and rounded down to 15
    minutes. Regions without a dish name are stored with ``needs_review``.
    """
    observed = round_to_quarter_hour(to_local_naive(plate.observed_at, timezone))
    meal = plate.meal.strip()
    session.add(
        Plate(
            plate_id=plate.plate_id,
            hall=hall,
            meal=meal,
            observed_at=observed,
            data_source=data_source,
        )
    )
    for region in plate.regions:
        dish = (region.dish_name or "").strip() or None
        session.add(
            PlateEvent(
                plate_id=plate.plate_id,
                hall=hall,
                meal=meal,
                observed_at=observed,
                dish_name=dish,
                category=region.category.value,
                volume_ml=region.volume_ml,
                grams=region.grams,
                grams_low=region.grams_low,
                grams_high=region.grams_high,
                percent_left=region.percent_left,
                match_score=region.match_score,
                needs_review=dish is None,
                data_source=data_source,
            )
        )


MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def build_router(
    db: Database,
    *,
    hall: str,
    timezone: str,
    require_device: Callable[..., None],
    pipeline: Callable[[], PlatePipeline] | None = None,
) -> APIRouter:
    """``pipeline`` builds (or returns the cached) vision pipeline; it may raise
    FileNotFoundError when the model is not installed, which becomes a 503."""
    router = APIRouter(tags=["plates"], dependencies=[Depends(require_device)])

    def store(plate: PlateIn) -> PlateAccepted:
        with Session(db.engine) as session:
            if session.exec(select(Plate).where(Plate.plate_id == plate.plate_id)).first():
                return _accepted(session, plate.plate_id)
            add_plate(session, plate, hall=hall, timezone=timezone)
            try:
                session.commit()
            except IntegrityError:  # the same plate arrived twice at once
                session.rollback()
            return _accepted(session, plate.plate_id)

    @router.post("/plates")
    def upload_plate(plate: PlateIn) -> PlateAccepted:
        """Store one plate measured by the rig.

        Idempotent on ``plate_id``: re-sending a plate the server already has
        stores nothing new and returns the counts stored the first time.
        """
        return store(plate)

    @router.post("/plates/image")
    def upload_plate_image(
        image: Annotated[UploadFile, File(description="JPEG or PNG, plate seen from above")],
        meal: Annotated[str, Form(min_length=1)],
    ) -> PlateImageAccepted:
        """Photo upload from the phone ``/capture`` page: segment, measure, match, store.

        Nothing is stored when the photo cannot be measured (422 with the reason). A
        photo with a plate and no food is stored as a clean plate.
        """
        if pipeline is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "vision pipeline disabled")
        data = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "image over 15 MB")
        try:
            frame = decode_image(data)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            runner = pipeline()
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        try:
            regions = runner.run(frame, meal.strip())
        except Unmeasurable as exc:
            raise HTTPException(422, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(422, f"meal {meal!r} is not on today's menu") from exc
        except MenuUnavailableError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"today's menu is unavailable: {exc}"
            ) from exc
        plate = PlateIn(
            plate_id=secrets.token_hex(8),
            meal=meal.strip(),
            observed_at=datetime.now(UTC),
            regions=regions,
        )
        accepted = store(plate)
        return PlateImageAccepted(**accepted.model_dump(), regions=regions)

    return router
