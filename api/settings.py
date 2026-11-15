"""Runtime configuration, read from environment variables (see ``.env.example``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    hall: str
    database_url: str
    data_dir: Path
    device_token: str
    """Bearer token the rig must present. Empty means uploads are refused."""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            hall=os.environ.get("LEFTOVER_HALL", "university-village"),
            database_url=os.environ.get("LEFTOVER_DATABASE_URL", "sqlite:///./leftover_lens.db"),
            data_dir=Path(os.environ.get("LEFTOVER_DATA_DIR", "./data")),
            device_token=os.environ.get("LEFTOVER_DEVICE_TOKEN", ""),
        )
