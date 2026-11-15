"""Runtime configuration, read from environment variables (see ``.env.example``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from analytics.assumptions import CostFactors

DEFAULT_CORS_ORIGINS = ("http://localhost:3000",)


@dataclass(frozen=True, slots=True)
class Settings:
    hall: str
    database_url: str
    data_dir: Path
    device_token: str
    """Bearer token the rig must present. Empty means uploads are refused."""
    staff_token: str = ""
    """Bearer token the staff dashboard must present. Empty means dashboard reads are refused."""
    timezone: str = "America/Los_Angeles"
    """The hall's local time zone: days, meals and weeks are counted in it."""
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    """Browser origins allowed to call the API (the Next.js dashboard)."""
    cost_factors: CostFactors = field(default_factory=CostFactors)
    seg_model: Path = Path("models/village-v2.onnx")
    """ONNX segmentation model for ``POST /plates/image``."""
    focal_px: float | None = None
    """Calibrated focal length of the rig camera in frame pixels; None uses the depth model's."""
    menu_date: date | None = None
    """Match photos against this day's menu instead of today's (replaying old photos in a demo)."""

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.environ.get("LEFTOVER_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))
        return cls(
            hall=os.environ.get("LEFTOVER_HALL", "university-village"),
            database_url=os.environ.get("LEFTOVER_DATABASE_URL", "sqlite:///./leftover_lens.db"),
            data_dir=Path(os.environ.get("LEFTOVER_DATA_DIR", "./data")),
            device_token=os.environ.get("LEFTOVER_DEVICE_TOKEN", ""),
            staff_token=os.environ.get("LEFTOVER_STAFF_TOKEN", ""),
            timezone=os.environ.get("LEFTOVER_TIMEZONE", "America/Los_Angeles"),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            cost_factors=CostFactors.from_env(),
            seg_model=Path(os.environ.get("LEFTOVER_SEG_MODEL", "models/village-v2.onnx")),
            focal_px=float(os.environ["LEFTOVER_FOCAL_PX"])
            if os.environ.get("LEFTOVER_FOCAL_PX")
            else None,
            menu_date=date.fromisoformat(os.environ["LEFTOVER_MENU_DATE"])
            if os.environ.get("LEFTOVER_MENU_DATE")
            else None,
        )
