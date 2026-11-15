from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from menu.importer import (
    MenuFetchError,
    MenuFormatError,
    MenuUnavailableError,
    build_url,
    cache_path,
    fetch_menu,
    get_menu,
    manual_path,
    parse_menu,
)

DAY = date(2026, 11, 14)
HALL = "university-village"


def client_returning(response: httpx.Response) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def failing_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_build_url_matches_the_documented_endpoint() -> None:
    assert build_url(date(2026, 11, 4)) == (
        "https://hospitality.usc.edu/wp-json/hsp-api/v1/get-res-dining-menus/"
        "university-village?y=2026&m=11&d=4"
    )


def test_parse_real_sample(village_payload: dict[str, Any]) -> None:
    menu = parse_menu(village_payload, expected_day=DAY)
    assert [m.name for m in menu.meals] == ["Breakfast", "Brunch", "Lunch", "Dinner"]
    assert menu.entry_count("Brunch") == 72
    assert menu.entry_count("Dinner") == 51
    # Weekend: breakfast and lunch are listed with no stations key at all.
    assert menu.entry_count("Breakfast") == 0
    burger = menu.meals[1].stations[0].menu[1]
    assert burger.item == "Boca Burger on White Bun"
    assert burger.allergens == ["sesame", "soy", "gluten"]
    assert burger.preferences == ["halal-ingredients", "vegan"]


def test_parse_rejects_wrong_day(village_payload: dict[str, Any]) -> None:
    with pytest.raises(MenuFormatError, match="asked for 2026-11-15"):
        parse_menu(village_payload, expected_day=date(2026, 11, 15))


@pytest.mark.parametrize("payload", [[], "closed", {"meals": []}, {"date": "20261114", "meals": 3}])
def test_parse_rejects_wrong_shape(payload: object) -> None:
    with pytest.raises(MenuFormatError):
        parse_menu(payload)


def test_parse_tolerates_non_list_tags() -> None:
    payload = {
        "date": "20261114",
        "meals": [
            {
                "name": "Dinner",
                "stations": [
                    {
                        "station": "Grill",
                        "menu": [{"item": "Rice", "allergens": False, "preferences": ""}],
                    }
                ],
            }
        ],
    }
    item = parse_menu(payload).meals[0].stations[0].menu[0]
    assert item.allergens == [] and item.preferences == []


def test_fetch_menu_calls_the_api(village_payload: dict[str, Any]) -> None:
    client, seen = client_returning(httpx.Response(200, json=village_payload))
    menu, body = fetch_menu(DAY, client=client)
    assert str(seen[0].url) == build_url(DAY)
    assert menu.entry_count("Brunch") == 72
    assert json.loads(body) == village_payload


def test_fetch_menu_http_error() -> None:
    client, _ = client_returning(httpx.Response(503))
    with pytest.raises(MenuFetchError):
        fetch_menu(DAY, client=client)


def test_fetch_menu_non_json() -> None:
    client, _ = client_returning(httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(MenuFormatError):
        fetch_menu(DAY, client=client)


def test_get_menu_caches_then_survives_an_outage(
    village_payload: dict[str, Any], tmp_path: Path
) -> None:
    client, _ = client_returning(httpx.Response(200, json=village_payload))
    _, source = get_menu(DAY, cache_dir=tmp_path, client=client)
    assert source == "api"
    assert cache_path(tmp_path, HALL, DAY).exists()

    menu, source = get_menu(DAY, cache_dir=tmp_path, client=failing_client())
    assert source == "cache"
    assert menu.entry_count("Dinner") == 51


def test_get_menu_falls_back_to_manual_file(
    village_payload: dict[str, Any], tmp_path: Path
) -> None:
    manual = manual_path(tmp_path, HALL, DAY)
    manual.parent.mkdir(parents=True)
    manual.write_text(json.dumps(village_payload), encoding="utf-8")
    _, source = get_menu(DAY, cache_dir=tmp_path, client=failing_client())
    assert source == "manual"


def test_get_menu_skips_a_corrupt_cache_file(
    village_payload: dict[str, Any], tmp_path: Path
) -> None:
    cached = cache_path(tmp_path, HALL, DAY)
    cached.parent.mkdir(parents=True)
    cached.write_text("{not json", encoding="utf-8")
    manual = manual_path(tmp_path, HALL, DAY)
    manual.parent.mkdir(parents=True)
    manual.write_text(json.dumps(village_payload), encoding="utf-8")
    _, source = get_menu(DAY, cache_dir=tmp_path, client=failing_client())
    assert source == "manual"


def test_get_menu_raises_when_nothing_is_available(tmp_path: Path) -> None:
    with pytest.raises(MenuUnavailableError):
        get_menu(DAY, cache_dir=tmp_path, client=failing_client())


def test_bad_api_response_does_not_overwrite_a_good_cache(
    village_payload: dict[str, Any], tmp_path: Path
) -> None:
    good, _ = client_returning(httpx.Response(200, json=village_payload))
    get_menu(DAY, cache_dir=tmp_path, client=good)
    bad, _ = client_returning(httpx.Response(200, json={"error": "nope"}))
    menu, source = get_menu(DAY, cache_dir=tmp_path, client=bad)
    assert source == "cache"
    assert menu.entry_count("Brunch") == 72
