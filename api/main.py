"""Application entry point: ``uvicorn api.main:app``."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api import dashboard, plates
from api.auth import bearer_guard
from api.models import Database
from api.pipeline import PlatePipeline, build_pipeline
from api.settings import Settings
from menu import load_daily_menu
from menu.importer import MenuUnavailableError
from menu.models import DailyMenu


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: Callable[[], PlatePipeline] | None = None,
) -> FastAPI:
    """Build the app. Tests pass their own ``Settings`` (and may inject a fake vision
    ``pipeline``); production reads the environment and loads the real models lazily."""
    cfg = settings or Settings.from_env()
    app = FastAPI(title="Leftover Lens API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    db = Database(cfg.database_url)
    app.state.db = db
    app.state.settings = cfg

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

    built: list[PlatePipeline] = []

    def vision_pipeline() -> PlatePipeline:
        """Load the models on the first photo, not at startup (they are large)."""
        if not built:
            built.append(
                build_pipeline(
                    cfg.seg_model,
                    hall=cfg.hall,
                    data_dir=cfg.data_dir,
                    today=lambda: datetime.now(ZoneInfo(cfg.timezone)).date(),
                    focal_px=cfg.focal_px,
                )
            )
        return built[0]

    app.include_router(
        plates.build_router(
            db,
            hall=cfg.hall,
            timezone=cfg.timezone,
            require_device=bearer_guard(cfg.device_token, "device"),
            pipeline=pipeline or vision_pipeline,
        )
    )
    app.include_router(
        dashboard.build_router(
            db,
            hall=cfg.hall,
            timezone=cfg.timezone,
            factors=cfg.cost_factors,
            require_staff=bearer_guard(cfg.staff_token, "staff"),
        )
    )
    return app


app = create_app()
