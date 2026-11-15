from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.models import round_to_quarter_hour
from api.settings import Settings
from menu.importer import MenuFetchError, cache_path

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path: Path, village_payload: dict[str, Any]) -> TestClient:
    # Pre-seed the menu cache; get_menu falls back to it when the live call fails.
    from datetime import date

    cached = cache_path(tmp_path / "menus", "test-hall", date(2026, 11, 14))
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps(village_payload), encoding="utf-8")
    settings = Settings(
        hall="test-hall", database_url="sqlite://", data_dir=tmp_path, device_token=TOKEN
    )
    return TestClient(create_app(settings))


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the network: the live menu call always fails."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise MenuFetchError("offline in tests")

    monkeypatch.setattr("menu.importer.fetch_menu", refuse)


def test_menu_endpoint_serves_the_cleaned_menu(client: TestClient) -> None:
    body = client.get("/menu/2026-11-14").json()
    assert body["source"] == "cache"
    assert [m["name"] for m in body["meals"]] == ["Brunch", "Dinner"]
    dinner = {i["name"]: i for i in body["meals"][1]["items"]}
    assert dinner["Rice Pilaf"]["category"] == "rice_grains"


def test_menu_endpoint_reports_a_missing_menu(client: TestClient) -> None:
    assert client.get("/menu/2026-11-15").status_code == 503
    assert client.get("/menu/not-a-date").status_code == 422


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "hall": "test-hall"}


def test_plates_requires_the_device_token(client: TestClient) -> None:
    assert client.post("/plates").status_code == 401
    assert client.post("/plates", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_plates_refuses_everything_when_no_token_is_configured(tmp_path: Path) -> None:
    settings = Settings(hall="h", database_url="sqlite://", data_dir=tmp_path, device_token="")
    app = TestClient(create_app(settings))
    assert app.post("/plates", headers={"Authorization": "Bearer "}).status_code == 401


def test_plates_is_not_implemented_yet(client: TestClient) -> None:
    response = client.post("/plates", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 501


def test_timestamps_round_down_to_15_minutes() -> None:
    assert round_to_quarter_hour(datetime(2026, 11, 14, 18, 44, 59, 5)) == datetime(
        2026, 11, 14, 18, 30
    )
    assert round_to_quarter_hour(datetime(2026, 11, 14, 18, 0)) == datetime(2026, 11, 14, 18, 0)
