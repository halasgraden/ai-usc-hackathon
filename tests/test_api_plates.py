"""POST /plates and POST /plates/image against a temporary SQLite file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select

from api.main import create_app
from api.models import Plate, PlateEvent
from api.settings import Settings

DEVICE = {"Authorization": "Bearer device-secret"}


@pytest.fixture
def app_and_client(tmp_path: Path) -> tuple[Any, TestClient]:
    settings = Settings(
        hall="test-hall",
        database_url=f"sqlite:///{tmp_path / 'plates.db'}",
        data_dir=tmp_path,
        device_token="device-secret",
        staff_token="staff-secret",
    )
    app = create_app(settings)
    return app, TestClient(app)


@pytest.fixture
def client(app_and_client: tuple[Any, TestClient]) -> TestClient:
    return app_and_client[1]


def region(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "category": "rice_grains", "dish_name": "Rice Pilaf", "match_score": 0.8,
        "volume_ml": 100.0, "grams": 73.0, "grams_low": 62.0, "grams_high": 84.0, "percent_left": 40.0,
    }  # fmt: skip
    return base | changes


def plate(*regions: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {"plate_id": "a1b2c3d4e5f6", "meal": "Dinner", "observed_at": "2026-11-14T18:07:31",
            "regions": list(regions)} | changes  # fmt: skip


def stored(app: Any) -> tuple[list[Plate], list[PlateEvent]]:
    with Session(app.state.db.engine) as session:
        return list(session.exec(select(Plate)).all()), list(session.exec(select(PlateEvent)).all())


def test_a_plate_is_stored_and_unknown_regions_go_to_review(
    app_and_client: tuple[Any, TestClient],
) -> None:
    app, client = app_and_client
    body = plate(region(), region(dish_name=None, match_score=None, percent_left=None))
    response = client.post("/plates", json=body, headers=DEVICE)
    assert response.status_code == 200
    assert response.json() == {
        "plate_id": "a1b2c3d4e5f6",
        "regions_stored": 2,
        "regions_for_review": 1,
    }
    plates, events = stored(app)
    assert [(p.hall, p.meal, p.data_source) for p in plates] == [("test-hall", "Dinner", "live")]
    assert {e.needs_review for e in events} == {True, False}
    assert all(e.observed_at.minute == 0 and e.observed_at.second == 0 for e in events)
    matched = next(e for e in events if not e.needs_review)
    assert (matched.dish_name, matched.category, matched.grams) == (
        "Rice Pilaf",
        "rice_grains",
        73.0,
    )


def test_a_blank_dish_name_counts_as_unknown(client: TestClient) -> None:
    response = client.post("/plates", json=plate(region(dish_name="  ")), headers=DEVICE)
    assert response.json()["regions_for_review"] == 1


def test_a_clean_plate_is_still_counted(app_and_client: tuple[Any, TestClient]) -> None:
    app, client = app_and_client
    response = client.post("/plates", json=plate(), headers=DEVICE)
    assert response.json() == {
        "plate_id": "a1b2c3d4e5f6",
        "regions_stored": 0,
        "regions_for_review": 0,
    }
    plates, events = stored(app)
    assert len(plates) == 1
    assert events == []


def test_resending_a_plate_stores_nothing_new(app_and_client: tuple[Any, TestClient]) -> None:
    app, client = app_and_client
    first = client.post("/plates", json=plate(region(), region()), headers=DEVICE).json()
    again = client.post("/plates", json=plate(region()), headers=DEVICE).json()
    assert (
        first == again == {"plate_id": "a1b2c3d4e5f6", "regions_stored": 2, "regions_for_review": 0}
    )
    plates, events = stored(app)
    assert (len(plates), len(events)) == (1, 2)


def test_aware_timestamps_become_hall_local_and_round_down(
    app_and_client: tuple[Any, TestClient],
) -> None:
    app, client = app_and_client
    # 02:07 UTC on 15 Nov is 18:07 Pacific (PST) on 14 Nov.
    client.post("/plates", json=plate(region(), observed_at="2026-11-15T02:07:00Z"), headers=DEVICE)
    plates, _ = stored(app)
    assert plates[0].observed_at.isoformat() == "2026-11-14T18:00:00"


@pytest.mark.parametrize(
    "bad",
    [
        plate(region(student_id="123")),
        plate(region(), device_owner="x"),
        plate(region(grams_low=90.0)),
        plate(region(), plate_id="short"),
    ],
)
def test_invalid_plates_are_rejected(client: TestClient, bad: dict[str, Any]) -> None:
    assert client.post("/plates", json=bad, headers=DEVICE).status_code == 422


def test_plates_need_the_device_token_not_the_staff_token(client: TestClient) -> None:
    body = plate(region())
    assert client.post("/plates", json=body).status_code == 401
    staff = {"Authorization": "Bearer staff-secret"}
    assert client.post("/plates", json=body, headers=staff).status_code == 401


def test_image_upload_needs_the_device_token(client: TestClient) -> None:
    assert client.post("/plates/image").status_code == 401


def test_tables_have_no_identifying_columns() -> None:
    forbidden = ("student", "card", "owner", "face", "device", "email", "name_of", "user")
    for table in SQLModel.metadata.sorted_tables:
        for column in table.columns:
            assert not any(word in column.name for word in forbidden), column.name


def test_importing_the_app_creates_no_database_file(tmp_path: Path) -> None:
    db_file = tmp_path / "lazy.db"
    create_app(Settings("h", f"sqlite:///{db_file}", tmp_path, "t"))
    assert not db_file.exists()


def test_openapi_lists_every_endpoint(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/plates", "/plates/image", "/dashboard/today", "/dashboard/dishes",
                 "/dashboard/dishes/{dish_name}", "/dashboard/trends", "/report/weekly"):  # fmt: skip
        assert path in paths


def test_cors_allows_the_dashboard_origin(client: TestClient) -> None:
    response = client.options(
        "/dashboard/today",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
