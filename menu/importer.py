"""Import a hall's daily menu from USC Hospitality's JSON endpoint.

This is the same JSON the public menu page loads. It is not a documented API, so
every successful response is cached and the importer degrades in order:

1. live API
2. the cached response for the same hall and day
3. a hand-entered file ``<cache_dir>/manual/<hall>/<YYYY-MM-DD>.json`` in the
   same format

We call it at most a few times a day (5 am and before each meal period).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import ValidationError

from menu.models import RawMenu

logger = logging.getLogger(__name__)

BASE_URL = "https://hospitality.usc.edu/wp-json/hsp-api/v1/get-res-dining-menus"
DEFAULT_HALL = "university-village"
USER_AGENT = "leftover-lens/0.1 (USC student hackathon project)"

MenuSource = Literal["api", "cache", "manual"]


class MenuError(Exception):
    """Base class for menu import failures."""


class MenuFetchError(MenuError):
    """The API could not be reached or answered with an error status."""


class MenuFormatError(MenuError):
    """The payload is not a menu in the format we understand."""


class MenuUnavailableError(MenuError):
    """No menu from the API, the cache or a manual file."""


def build_url(day: date, hall: str = DEFAULT_HALL) -> str:
    """URL of the menu for ``hall`` on ``day`` (month and day are not zero-padded)."""
    return f"{BASE_URL}/{hall}?y={day.year}&m={day.month}&d={day.day}"


def parse_menu(payload: object, *, expected_day: date | None = None) -> RawMenu:
    """Validate a decoded JSON payload.

    Args:
        payload: the decoded JSON document.
        expected_day: if given, the payload's own date must match it, which guards
            against the endpoint silently answering for a different day.

    Raises:
        MenuFormatError: wrong shape, unreadable date, or a date mismatch.
    """
    try:
        menu = RawMenu.model_validate(payload)
    except ValidationError as exc:
        raise MenuFormatError(f"unexpected menu format: {exc}") from exc
    if expected_day is not None:
        try:
            served = datetime.strptime(menu.date, "%Y%m%d").date()
        except ValueError as exc:
            raise MenuFormatError(f"unreadable menu date {menu.date!r}") from exc
        if served != expected_day:
            raise MenuFormatError(f"asked for {expected_day}, got a menu for {served}")
    return menu


def fetch_menu(
    day: date,
    hall: str = DEFAULT_HALL,
    *,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
) -> tuple[RawMenu, str]:
    """Fetch one day's menu from the live API.

    Returns:
        The parsed menu and the response body, so the caller can cache it verbatim.

    Raises:
        MenuFetchError: network failure, timeout or non-2xx status.
        MenuFormatError: the response is not valid menu JSON for ``day``.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
    try:
        response = http.get(build_url(day, hall))
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MenuFetchError(f"menu API request failed: {exc}") from exc
    finally:
        if owns_client:
            http.close()
    try:
        payload = response.json()
    except ValueError as exc:
        raise MenuFormatError("menu API did not return JSON") from exc
    return parse_menu(payload, expected_day=day), response.text


def cache_path(cache_dir: Path, hall: str, day: date) -> Path:
    """Where the raw response for ``hall`` on ``day`` is cached."""
    return cache_dir / hall / f"{day.isoformat()}.json"


def manual_path(cache_dir: Path, hall: str, day: date) -> Path:
    """Where a hand-entered fallback menu for ``hall`` on ``day`` is looked for."""
    return cache_dir / "manual" / hall / f"{day.isoformat()}.json"


def load_menu_file(path: Path, *, expected_day: date | None = None) -> RawMenu:
    """Read a saved menu JSON file (a cached response, a fixture or a manual menu).

    Raises:
        MenuFormatError: the file is not valid menu JSON.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise MenuFormatError(f"{path} is not valid JSON") from exc
    return parse_menu(payload, expected_day=expected_day)


def get_menu(
    day: date,
    *,
    hall: str = DEFAULT_HALL,
    cache_dir: Path,
    client: httpx.Client | None = None,
) -> tuple[RawMenu, MenuSource]:
    """Get the menu for ``day``: live API first, then cache, then a manual file.

    A successful API response is written to the cache before returning.

    Raises:
        MenuUnavailableError: all three sources failed.
    """
    cached = cache_path(cache_dir, hall, day)
    try:
        menu, body = fetch_menu(day, hall, client=client)
    except MenuError as exc:
        logger.warning("live menu unavailable for %s %s: %s", hall, day, exc)
    else:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(body, encoding="utf-8")
        return menu, "api"

    fallbacks: list[tuple[Path, MenuSource]] = [
        (cached, "cache"),
        (manual_path(cache_dir, hall, day), "manual"),
    ]
    for path, source in fallbacks:
        if not path.exists():
            continue
        try:
            return load_menu_file(path, expected_day=day), source
        except MenuFormatError as exc:
            logger.warning("ignoring unreadable %s menu file %s: %s", source, path, exc)
    raise MenuUnavailableError(f"no menu for {hall} on {day} from the API, cache or manual file")
