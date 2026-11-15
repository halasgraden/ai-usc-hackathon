"""Application entry point: ``uvicorn api.main:app``."""

from __future__ import annotations

import hmac
from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status

from api.settings import Settings
from menu import load_daily_menu
from menu.importer import MenuUnavailableError
from menu.models import DailyMenu


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own ``Settings``; production reads the environment."""
    cfg = settings or Settings.from_env()
    app = FastAPI(title="Leftover Lens API", version="0.1.0")

    def require_device(authorization: Annotated[str | None, Header()] = None) -> None:
        """Only the rig (or the phone capture page) may upload plates."""
        expected = f"Bearer {cfg.device_token}"
        supplied = authorization or ""
        if not cfg.device_token or not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid device token")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "hall": cfg.hall}

    @app.get("/menu/{day}")
    def menu_for_day(day: date) -> DailyMenu:
        """The cleaned, categorized menu: what the camera may recognize that day."""
        try:
            return load_daily_menu(day, hall=cfg.hall, data_dir=cfg.data_dir)
        except MenuUnavailableError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    @app.post("/plates", dependencies=[Depends(require_device)])
    def upload_plate() -> None:
        """Receive one plate (image from the phone, or measurements from the rig)."""
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "plate ingestion lands in hour 1-2")

    return app


app = create_app()
