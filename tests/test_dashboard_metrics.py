"""End-to-end metric contracts for the full-menu dashboard."""

from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from api.main import create_app
from api.metrics import comparison
from api.models import Database, Plate, PlateEvent, ServingRecord
from api.seed import seed
from tests.test_api_dashboard import STAFF, post, region, settings_for


@pytest.fixture
def demo(tmp_path: Path) -> tuple[TestClient, Database]:
    settings = settings_for(tmp_path)
    db = Database(settings.database_url)
    seed(
        db,
        hall=settings.hall,
        timezone=settings.timezone,
        start=date(2026, 10, 19),
        end=date(2026, 11, 15),
    )
    return TestClient(create_app(settings)), db


def test_full_menu_ranges_serving_weights_and_report_agree(
    demo: tuple[TestClient, Database],
) -> None:
    client, db = demo
    params = "start=2026-11-02&end=2026-11-08"
    data = client.get("/dashboard/dishes?" + params, headers=STAFF).json()
    assert len(data["rows"]) == 37
    assert data["data_source"] == "seeded"
    total = sum(d["total_kg"] for d in data["rows"]) + data["unmatched_kg"]
    assert total == pytest.approx(data["headline"]["kg_wasted"], abs=0.0001)
    for row in data["rows"]:
        assert row["kg_low"] <= row["total_kg"] <= row["kg_high"]
        assert row["mean_grams"] == pytest.approx(
            row["total_kg"] * 1000 / row["plates_seen"], abs=0.01
        )
        serving = row["serving_comparison"]
        assert (
            serving["returned_percent_low"]
            <= serving["returned_percent"]
            <= serving["returned_percent_high"]
            <= 100
        )
        assert serving["returned_percent"] == pytest.approx(
            100 * row["total_kg"] / serving["served_kg"], abs=0.001
        )
    h = data["headline"]["serving_comparison"]
    assert h["returned_percent"] == pytest.approx(
        100 * sum(d["total_kg"] for d in data["rows"]) / h["served_kg"], abs=0.001
    )
    report = client.get("/report/weekly?week_start=2026-11-02", headers=STAFF).json()
    assert report["rows"] == data["rows"]
    assert report["headline"] == data["headline"]
    assert report["plates_seen"] == data["plates_seen"]
    with Session(db.engine) as session:
        assert all(r.capture_complete for r in session.exec(select(ServingRecord)).all())
        plates = session.exec(select(Plate)).all()
        events = session.exec(select(PlateEvent)).all()
        assert len(plates) > len({e.plate_id for e in events})  # clean plates retained


def test_filters_match_across_summary_trends_detail_and_report(
    demo: tuple[TestClient, Database],
) -> None:
    client, _ = demo
    filters = "meal=Dinner&station=Flexitarian&search=rice"
    data = client.get(
        "/dashboard/dishes?start=2026-11-02&end=2026-11-08&" + filters, headers=STAFF
    ).json()
    assert [r["dish_name"] for r in data["rows"]] == ["Rice Pilaf"]
    trends = client.get(
        "/dashboard/trends?start=2026-11-02&end=2026-11-08&" + filters, headers=STAFF
    ).json()
    report = client.get("/report/weekly?week_start=2026-11-02&" + filters, headers=STAFF).json()
    assert trends["weeks"][0]["kg_wasted"] == data["headline"]["kg_wasted"]
    assert len(trends["dishes"]) == 1
    assert trends["weeks"][0]["partial"] is False
    assert report["rows"] == data["rows"]
    detail = client.get(
        "/dashboard/dishes/Rice%20Pilaf?start=2026-11-02&end=2026-11-08&meal=Dinner", headers=STAFF
    ).json()
    assert detail["dish"] == data["rows"][0]


def test_no_serving_comparison_for_live_or_incomplete_records(
    demo: tuple[TestClient, Database],
) -> None:
    client, db = demo
    post(
        client,
        "live-mixed-plate",
        "Dinner",
        datetime(2026, 11, 8, 18),
        region("Rice Pilaf", 100, 20),
    )
    data = client.get("/dashboard/dishes?start=2026-11-02&end=2026-11-08", headers=STAFF).json()
    rice = next(r for r in data["rows"] if r["dish_name"] == "Rice Pilaf")
    assert rice["serving_comparison"] is None
    assert data["headline"]["serving_comparison"] is None
    assert data["data_source"] == "seeded"
    with Session(db.engine) as session:
        records = session.exec(select(ServingRecord)).all()
        records[0].capture_complete = False
        assert comparison([], records) is None
    assert comparison([], []) is None


def test_missing_meal_denominator_rejected(demo: tuple[TestClient, Database]) -> None:
    _, db = demo
    with Session(db.engine) as session:
        events = [e for e in session.exec(select(PlateEvent)).all() if e.dish_name == "Rice Pilaf"]
        records = [
            r for r in session.exec(select(ServingRecord)).all() if r.dish_name == "Rice Pilaf"
        ]
        assert comparison(events, records) is not None
        assert comparison(events, records[1:]) is None


def test_partial_windows_and_empty_periods(demo: tuple[TestClient, Database]) -> None:
    client, _ = demo
    data = client.get("/dashboard/trends?start=2026-11-03&end=2026-11-05", headers=STAFF).json()
    assert data["weeks"][0]["partial"] is True
    empty = client.get("/dashboard/trends?start=2025-01-01&end=2025-01-31", headers=STAFF).json()
    assert empty["weeks"] == []
    assert empty["dishes"] == []
    assert empty["data_source"] == "seeded"
