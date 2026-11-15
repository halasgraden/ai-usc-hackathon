"""Menu-first matching, stage 3 of the pipeline: import, clean, categorize.

Typical use::

    from datetime import date
    from pathlib import Path
    from menu import load_daily_menu

    menu = load_daily_menu(date.today(), data_dir=Path("data"))
    candidates = menu.meal("Dinner").candidates()   # what a plate may contain
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx

from menu.categorizer import Categorizer, CategoryCache
from menu.cleaner import CleaningRules, clean_menu
from menu.importer import DEFAULT_HALL, get_menu
from menu.models import DailyMenu

__all__ = ["DailyMenu", "load_daily_menu"]


def load_daily_menu(
    day: date,
    *,
    hall: str = DEFAULT_HALL,
    data_dir: Path = Path("data"),
    client: httpx.Client | None = None,
) -> DailyMenu:
    """Fetch (or load from cache), clean and categorize one hall's menu for ``day``.

    Menus are cached under ``data_dir/menus/<hall>/`` and category decisions in
    ``data_dir/category_cache.json``.

    Raises:
        menu.importer.MenuUnavailableError: no API response, cache or manual file.
    """
    raw, source = get_menu(day, hall=hall, cache_dir=data_dir / "menus", client=client)
    cleaned = clean_menu(raw, CleaningRules.load(), hall=hall, source=source)
    categorizer = Categorizer.load(cache=CategoryCache(data_dir / "category_cache.json"))
    return categorizer.apply(cleaned)
