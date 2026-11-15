"""Dashboard and report endpoints, and the seed script, against a temporary SQLite file."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, func, select

from analytics.assumptions import CostFactors
from api import seed as seed_module
from api.main import create_app
from api.models import Database, Plate, PlateEvent
from api.seed import clear_seeded, generate_plates, seed
from api.settings import Settings

DEVICE = {"Authorization": "Bearer device-secret"}
STAFF = {"Authorization": "Bearer staff-secret"}
FACTORS = CostFactors(cost_usd_per_kg=10.0, co2e_kg_per_kg=2.0)
DAY = date(2026, 11, 10)  # a Tuesday
STAFF_PATHS = (
    "/dashboard/today",
    "/dashboard/dishes",
    "/dashboard/dishes/Rice Pilaf",
    "/dashboard/trends",
)


def settings_for(tmp_path: Path, **changes: Any) -> Settings:
    base: dict[str, Any] = {
        "hall": "test-hall",
        "database_url": f"sqlite:///{tmp_path / 'dash.db'}",
        "data_dir": tmp_path,
        "device_token": "device-secret",
        "staff_token": "staff-secret",
        "cost_factors": FACTORS,
    }
    return Settings(**(base | changes))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(tmp_path)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def region(
    dish: str | None, grams: float, pct: float | None, category: str = "rice_grains"
) -> dict[str, Any]:
    return {"category": category, "dish_name": dish, "volume_ml": grams, "grams": grams,
            "grams_low": grams, "grams_high": grams, "percent_left": pct}  # fmt: skip


def post(
    client: TestClient, plate_id: str, meal: str, at: datetime, *regions: dict[str, Any]
) -> None:
    body = {
        "plate_id": plate_id,
        "meal": meal,
        "observed_at": at.isoformat(),
        "regions": list(regions),
    }
    assert client.post("/plates", json=body, headers=DEVICE).status_code == 200


@pytest.fixture
def live(client: TestClient) -> TestClient:
    """Four plates on DAY (one clean, one with an unknown region) and one the week before."""
    at = datetime.combine(DAY, datetime.min.time())
    post(client, "plate-aaaa", "Dinner", at.replace(hour=18, minute=5),
         region("Rice Pilaf", 100, 50), region("Garlic and Herb Roasted Chicken", 200, 20, "poultry"))  # fmt: skip
    post(client, "plate-bbbb", "Dinner", at.replace(hour=18, minute=20),
         region("Rice Pilaf", 300, 100), region(None, 80, None, "cooked_vegetables"))  # fmt: skip
    post(client, "plate-cccc", "Lunch", at.replace(hour=12), region("Rice Pilaf", 50, 10))
    post(client, "plate-dddd", "Dinner", at.replace(hour=19))
    post(client, "plate-eeee", "Dinner", at.replace(day=5, hour=18), region("Rice Pilaf", 100, 20))
    return client


# --- auth -------------------------------------------------------------------


@pytest.mark.parametrize("path", STAFF_PATHS)
def test_staff_endpoints_need_the_staff_token(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers=DEVICE).status_code == 401


@pytest.mark.parametrize("path", STAFF_PATHS)
def test_no_staff_token_configured_refuses_everyone(tmp_path: Path, path: str) -> None:
    client = TestClient(create_app(settings_for(tmp_path, staff_token="")))
    assert client.get(path, headers={"Authorization": "Bearer "}).status_code == 401


def test_the_weekly_report_is_public(client: TestClient) -> None:
    assert client.get("/report/weekly").status_code == 200


# --- today ------------------------------------------------------------------


def test_today_headline_and_top_dishes(live: TestClient) -> None:
    body = live.get(f"/dashboard/today?day={DAY}", headers=STAFF).json()
    assert body["hall"] == "test-hall"
    assert body["data_source"] == "live"
    assert body["meal"] is None
    assert body["plates_seen"] == 4  # the clean plate counts
    assert body["regions_for_review"] == 1
    assert body["headline"] == {
        "kg_wasted": 0.73,  # unknown region included
        "waste_rate_percent": 45.0,  # mean of 50, 20, 100, 10 (matched only)
        "est_cost_usd": 7.3,
        "est_co2e_kg": 1.46,
    }
    rice, chicken = body["top_dishes"]
    assert rice == {
        "dish_name": "Rice Pilaf", "station": "Flexitarian", "category": "rice_grains",
        "plates_seen": 3, "total_kg": 0.45, "mean_percent_left": 53.3, "trend": "up",
    }  # fmt: skip
    assert chicken["category"] == "poultry"
    assert chicken["trend"] == "unknown"  # not seen in the previous 7 days
    # Rule 3: the cost and CO2e estimates travel with the factors behind them.
    assert {"cost_usd_per_kg", "co2e_kg_per_kg"} <= set(body["assumptions"])
    assert "placeholder" in body["assumptions"]["cost_usd_per_kg"].lower()


def test_today_filters_by_meal_case_insensitively(live: TestClient) -> None:
    body = live.get(f"/dashboard/today?day={DAY}&meal=dinner", headers=STAFF).json()
    assert body["meal"] == "dinner"
    assert body["plates_seen"] == 3
    assert body["headline"]["kg_wasted"] == 0.68
    assert body["top_dishes"][0]["mean_percent_left"] == 75.0


def test_today_keeps_at_most_five_dishes(client: TestClient) -> None:
    at = datetime.combine(DAY, datetime.min.time()).replace(hour=12)
    post(
        client, "plate-many", "Lunch", at, *(region(f"Dish {i}", 10 * (i + 1), 5) for i in range(7))
    )
    rows = client.get(f"/dashboard/today?day={DAY}", headers=STAFF).json()["top_dishes"]
    assert [r["dish_name"] for r in rows] == [f"Dish {i}" for i in (6, 5, 4, 3, 2)]
    assert rows[0]["station"] is None  # not a known Village dish


def test_today_with_no_plates_is_empty_and_live(client: TestClient) -> None:
    body = client.get("/dashboard/today", headers=STAFF).json()
    assert body["data_source"] == "live"
    assert body["plates_seen"] == 0
    assert body["top_dishes"] == []
    assert body["headline"]["waste_rate_percent"] is None


# --- dishes -----------------------------------------------------------------


def test_dishes_table(live: TestClient) -> None:
    body = live.get(f"/dashboard/dishes?start=2026-11-08&end={DAY}", headers=STAFF).json()
    assert (body["start"], body["end"], body["data_source"]) == ("2026-11-08", str(DAY), "live")
    assert [r["dish_name"] for r in body["rows"]] == [
        "Rice Pilaf",
        "Garlic and Herb Roasted Chicken",
    ]
    # 53.3 now against 20 in the 3 days before (5-7 Nov)
    assert body["rows"][0]["trend"] == "up"
    one_day = live.get(f"/dashboard/dishes?start={DAY}&end={DAY}", headers=STAFF).json()
    assert one_day["rows"][0]["trend"] == "unknown"  # nothing on 9 Nov


def test_dishes_default_to_the_last_seven_days(live: TestClient) -> None:
    body = live.get(f"/dashboard/dishes?end={DAY}", headers=STAFF).json()
    assert body["start"] == str(DAY - timedelta(days=6))
    rice = body["rows"][0]
    assert (rice["plates_seen"], rice["total_kg"]) == (4, 0.55)


def test_dishes_reject_a_backwards_or_huge_range(client: TestClient) -> None:
    assert (
        client.get("/dashboard/dishes?start=2026-11-10&end=2026-11-01", headers=STAFF).status_code
        == 422
    )
    assert (
        client.get("/dashboard/dishes?start=2024-01-01&end=2026-11-01", headers=STAFF).status_code
        == 422
    )
    assert client.get("/dashboard/dishes?start=nope", headers=STAFF).status_code == 422


# --- dish detail ------------------------------------------------------------


def test_dish_detail(live: TestClient) -> None:
    body = live.get(
        f"/dashboard/dishes/Rice Pilaf?start=2026-11-01&end={DAY}", headers=STAFF
    ).json()
    assert body["data_source"] == "live"
    assert body["dish"]["plates_seen"] == 4
    assert body["percent_left_samples"] == [20.0, 10.0, 50.0, 100.0]  # in time order
    assert body["weekly"] == [
        {"week_start": "2026-11-02", "kg_wasted": 0.1, "waste_rate_percent": 20.0},
        {"week_start": "2026-11-09", "kg_wasted": 0.45, "waste_rate_percent": 53.3},
    ]


def test_dish_detail_handles_quotes_and_slashes(client: TestClient) -> None:
    at = datetime.combine(DAY, datetime.min.time()).replace(hour=12)
    post(client, "plate-slash", "Lunch", at, region("Mac / Cheese", 40, 30))
    post(client, "plate-quote", "Lunch", at, region('Portobello Red Wine "Pot Roast"', 40, 30))
    for name in (
        "Mac%20%2F%20Cheese",
        "Mac / Cheese",
        "Portobello%20Red%20Wine%20%22Pot%20Roast%22",
    ):
        response = client.get(f"/dashboard/dishes/{name}?end={DAY}", headers=STAFF)
        assert response.status_code == 200, name


def test_unknown_dish_is_404(live: TestClient) -> None:
    assert live.get(f"/dashboard/dishes/Nope?end={DAY}", headers=STAFF).status_code == 404


# --- trends -----------------------------------------------------------------


def test_trends_cover_recent_weeks_and_skip_empty_ones(client: TestClient) -> None:
    now = datetime.now(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None, hour=12, minute=0)
    post(client, "plate-now1", "Lunch", now, region("Rice Pilaf", 200, 40))
    post(client, "plate-old1", "Lunch", now - timedelta(days=7), region("Rice Pilaf", 100, 20))
    post(client, "plate-ancient", "Lunch", now - timedelta(weeks=20), region("Rice Pilaf", 100, 20))

    weeks = client.get("/dashboard/trends", headers=STAFF).json()["weeks"]
    assert [(w["kg_wasted"], w["waste_rate_percent"]) for w in weeks] == [(0.1, 20.0), (0.2, 40.0)]
    assert len(client.get("/dashboard/trends?weeks=1", headers=STAFF).json()["weeks"]) == 1
    assert client.get("/dashboard/trends?weeks=0", headers=STAFF).status_code == 422
    assert client.get("/dashboard/trends?weeks=53", headers=STAFF).status_code == 422


# --- weekly report ----------------------------------------------------------


def test_weekly_report(live: TestClient) -> None:
    body = live.get("/report/weekly?week_start=2026-11-12").json()  # a Thursday
    assert body["week_start"] == "2026-11-09"
    assert body["data_source"] == "live"
    assert body["headline"]["kg_wasted"] == 0.73
    assert body["headline"]["est_cost_usd"] == 7.3
    assert body["top_dishes"][0]["dish_name"] == "Rice Pilaf"
    assert body["top_dishes"][0]["trend"] == "up"
    assert set(body["assumptions"]) == {"cost_usd_per_kg", "co2e_kg_per_kg", "basis"}
    assert "10.00 USD" in body["assumptions"]["cost_usd_per_kg"]
    assert "index" in body["methods_note"]
    assert "simulated" not in body["methods_note"]


def test_weekly_report_never_lists_plates(live: TestClient) -> None:
    text = live.get("/report/weekly?week_start=2026-11-09").text
    assert "plate-aaaa" not in text
    assert "plate_id" not in text


# --- seeded data ------------------------------------------------------------


def test_generated_plates_are_deterministic_and_plausible() -> None:
    first = generate_plates(DAY, DAY + timedelta(days=1), seed=7)
    assert first == generate_plates(DAY, DAY + timedelta(days=1), seed=7)
    assert first != generate_plates(DAY, DAY + timedelta(days=1), seed=8)
    # a day is generated the same way whatever window it falls in
    assert [p for p in first if p.observed_at.date() == DAY] == generate_plates(DAY, DAY, seed=7)
    assert {p.meal for p in first} == {"Breakfast", "Lunch", "Dinner"}
    regions = [r for p in first for r in p.regions]
    assert all(r.match_score is None for r in regions)  # no invented model numbers
    assert 0 < sum(r.dish_name is None for r in regions) < len(regions) * 0.15
    weekend = generate_plates(date(2026, 11, 14), date(2026, 11, 14), seed=7)
    assert {p.meal for p in weekend} == {"Brunch", "Dinner"}


def count(db: Database, source: str) -> int:
    with Session(db.engine) as session:
        query = select(func.count()).select_from(Plate).where(Plate.data_source == source)
        return int(session.exec(query).one())


def test_seeded_responses_are_labelled(live: TestClient, settings: Settings) -> None:
    db = Database(settings.database_url)
    seed(db, hall="test-hall", timezone=settings.timezone, start=DAY, end=DAY)
    for path in (f"/dashboard/today?day={DAY}", f"/dashboard/dishes?end={DAY}",
                 f"/dashboard/dishes/Rice Pilaf?end={DAY}", "/dashboard/trends"):  # fmt: skip
        assert live.get(path, headers=STAFF).json()["data_source"] == "seeded", path
    report = live.get(f"/report/weekly?week_start={DAY}").json()
    assert report["data_source"] == "seeded"
    assert "simulated" in report["methods_note"]
    # an empty window in a database that holds seeded plates is still labelled
    assert (
        live.get("/dashboard/today?day=2020-01-01", headers=STAFF).json()["data_source"] == "seeded"
    )


def test_reseeding_replaces_seeded_plates_and_keeps_live_ones(
    live: TestClient, settings: Settings
) -> None:
    db = Database(settings.database_url)
    n = seed(db, hall="test-hall", timezone=settings.timezone, start=DAY, end=DAY)
    assert n > 100
    assert seed(db, hall="test-hall", timezone=settings.timezone, start=DAY, end=DAY) == n
    assert (count(db, "seeded"), count(db, "live")) == (n, 5)
    clear_seeded(db, "test-hall")
    assert (count(db, "seeded"), count(db, "live")) == (0, 5)
    with Session(db.engine) as session:
        assert {e.data_source for e in session.exec(select(PlateEvent)).all()} == {"live"}


def test_seed_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LEFTOVER_HALL", "test-hall")
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    seed_module.main(["--weeks", "1", "--end", "2026-11-14", "--database-url", url])
    assert "2026-11-08 to 2026-11-14" in capsys.readouterr().out
    db = Database(url)
    assert count(db, "seeded") > 1000
    seed_module.main(["--clear", "--database-url", url])
    assert count(db, "seeded") == 0
