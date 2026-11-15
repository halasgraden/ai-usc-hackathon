from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menu.categorizer import Categorizer
from menu.cleaner import CleaningRules, clean_menu
from menu.importer import parse_menu
from menu.models import DailyMenu, RawMenu

FIXTURES = Path(__file__).parent / "fixtures"
VILLAGE_SAMPLE = FIXTURES / "village_2026-11-14.json"


@pytest.fixture(scope="session")
def village_payload() -> dict[str, Any]:
    """The Village menu for 2026-11-14 as the API served it."""
    payload: dict[str, Any] = json.loads(VILLAGE_SAMPLE.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="session")
def raw_menu(village_payload: dict[str, Any]) -> RawMenu:
    return parse_menu(village_payload)


@pytest.fixture(scope="session")
def rules() -> CleaningRules:
    return CleaningRules.load()


@pytest.fixture(scope="session")
def cleaned(raw_menu: RawMenu, rules: CleaningRules) -> DailyMenu:
    return clean_menu(raw_menu, rules, hall="university-village")


@pytest.fixture(scope="session")
def categorized(cleaned: DailyMenu) -> DailyMenu:
    return Categorizer.load().apply(cleaned)
