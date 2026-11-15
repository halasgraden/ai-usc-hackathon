"""Command line: show what the camera will be allowed to recognize today.

    python -m menu                       # today's Village menu from the API
    python -m menu --date 2026-11-14
    python -m menu --file tests/fixtures/village_2026-11-14.json
    python -m menu --json > menu.json

A live fetch also caches the raw response under <data-dir>/menus/<hall>/, which
is how to save a new sample for tests.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from menu import load_daily_menu
from menu.categorizer import Categorizer
from menu.cleaner import CleaningRules, clean_menu
from menu.importer import DEFAULT_HALL, MenuError, load_menu_file
from menu.models import DailyMenu, ItemRole


def render(menu: DailyMenu) -> str:
    """Plain-text summary of a cleaned menu."""
    lines = [f"{menu.hall}  {menu.day.isoformat()}  (source: {menu.source})"]
    if not menu.meals:
        lines.append("No meals listed for this day.")
    for meal in menu.meals:
        roles = {role: sum(i.role is role for i in meal.items) for role in ItemRole}
        lines.append("")
        lines.append(
            f"{meal.name}: {len(meal.candidates())} plate candidates "
            f"({roles[ItemRole.COMPONENT]} components, {roles[ItemRole.CONDIMENT]} condiments, "
            f"{roles[ItemRole.NOTICE]} notices set aside)"
        )
        for item in meal.candidates():
            category = item.category.value if item.category else "UNCATEGORIZED"
            mark = " *" if item.synthetic else ""
            lines.append(f"  {item.name + mark:<52} {category:<24} {', '.join(item.stations)}")
        if meal.uncategorized():
            lines.append("  -> add the uncategorized dishes to menu/data/overrides.yaml")
    lines.append("")
    lines.append("* implied by the menu, added by us (see composites in cleaning_rules.yaml)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m menu", description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--hall", default=DEFAULT_HALL)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--file", type=Path, help="read a saved menu JSON instead of the API")
    parser.add_argument("--json", action="store_true", help="print the cleaned menu as JSON")
    args = parser.parse_args(argv)

    try:
        if args.file:
            raw = load_menu_file(args.file)
            cleaned = clean_menu(raw, CleaningRules.load(), hall=args.hall, source="file")
            menu = Categorizer.load().apply(cleaned)
        else:
            menu = load_daily_menu(args.date, hall=args.hall, data_dir=args.data_dir)
    except (MenuError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(menu.model_dump_json(indent=1) if args.json else render(menu))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
