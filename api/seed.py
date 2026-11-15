"""Deterministic demo data: a few weeks of plausible plates for USC Village.

Run from the repo root::

    python -m api.seed                       # 4 weeks ending today, into LEFTOVER_DATABASE_URL
    python -m api.seed --weeks 6 --end 2026-11-14
    python -m api.seed --clear               # remove seeded plates only

Every plate written here is stored with ``data_source="seeded"``, so every API
response built from it says so (CONTRACTS rule 2). Re-running replaces earlier
seeded plates for the hall and never touches live ones.

The numbers are simulated, not measured: dish names are real (the Village menu
of 14 Nov 2026, see ``api/dishes.py``), but popularity, leftovers and the gentle
downward trend are invented for the demo. Each (day, meal) draws from its own
random stream keyed by ``--seed``, so a given day always gets the same plates.
No match scores are generated (they would look like model accuracy).
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, delete

from api.dishes import BRUNCH, DINNER, KnownDish
from api.models import Database, Plate, PlateEvent, ServingRecord
from api.plates import add_plate
from api.schemas import PlateIn, RegionIn
from api.settings import Settings
from core.types import FoodCategory as C
from portion.grams import estimate_grams

DEFAULT_SEED = 20261114
DEFAULT_WEEKS = 4

REFERENCE_ML: dict[C, float] = {
    C.RICE_GRAINS: 180, C.PASTA_PLAIN: 200, C.PASTA_SAUCY: 250, C.SOUP_STEW: 300,
    C.MASHED_PUREE: 200, C.BEANS_LENTILS: 150, C.POULTRY: 150, C.RED_MEAT: 150,
    C.FISH_SEAFOOD: 140, C.EGGS: 120, C.TOFU_PLANT_PROTEIN: 140, C.RAW_VEGETABLES: 100,
    C.COOKED_VEGETABLES: 120, C.LEAFY_SALAD: 150, C.POTATOES_FRIED_ROASTED: 150, C.BREAD: 80,
    C.PIZZA_BURGER_SANDWICH: 250, C.BAKED_DESSERT: 100, C.ICE_CREAM: 120,
    C.YOGURT_PUDDING: 150, C.CUT_FRUIT: 150,
}  # fmt: skip
"""Simulated reference portion (ml) per category; percent left is volume left against it."""

# dish name -> (relative popularity, typical percent left). Unlisted dishes: (1.0, 22).
PROFILE: dict[str, tuple[float, float]] = {
    "Assorted French Fries": (3.0, 18), "Beef Burger on Martin's Potato Bun": (2.2, 15),
    "Boca Burger on White Bun": (0.8, 24), "All Beef Hot Dog on White Bun": (1.0, 14),
    "Deli Sandwich": (1.6, 17), "Scrambled Eggs": (2.5, 26), "Bacon": (2.0, 9),
    "Crispy Fried Tater Tots": (2.2, 20), "Oatmeal": (1.0, 38), "Assorted Fruits": (1.8, 30),
    "Made-to-Order Omelette": (1.4, 12), "Chipotle Roasted Yams": (0.7, 35),
    "Roasted Broccoli": (1.0, 40), "Fried BBQ Tofu": (0.6, 33), "Rice Pilaf": (2.4, 34),
    "Garlic and Herb Roasted Chicken": (2.4, 21), "Braised Beef Short Ribs": (1.5, 19),
    "Braised Lentils with Garlic and Lemon": (0.7, 41), "Roasted Lemon Zest Asparagus": (0.9, 37),
    "Maple Roasted Carrots": (1.0, 31), "Baked Potatoes": (1.3, 29), "Salad Bar Salad": (2.6, 27),
    "Beef Chili with Beans": (1.1, 24), 'Portobello Red Wine "Pot Roast"': (0.6, 36),
    "Chicken Noodle Soup with Egg Noodles": (1.2, 22), "French Style Crepe": (1.2, 11),
    "Molly's Garden Vegetable Soup with Kidney Beans": (0.7, 32),
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class MealPlan:
    name: str
    opens: time
    closes: time
    plates: int
    dishes: tuple[KnownDish, ...]


WEEKDAY_MEALS = (
    MealPlan("Breakfast", time(7, 0), time(10, 30), 90, BRUNCH),
    MealPlan("Lunch", time(11, 0), time(16, 0), 240, DINNER),
    MealPlan("Dinner", time(16, 30), time(21, 0), 280, DINNER),
)
WEEKEND_MEALS = (
    MealPlan("Brunch", time(9, 30), time(15, 0), 200, BRUNCH),
    MealPlan("Dinner", time(16, 30), time(20, 0), 220, DINNER),
)
UNKNOWN_SHARE = 0.05
"""Share of regions the simulated matcher leaves unmatched (they go to review)."""
WEEKLY_IMPROVEMENT = 0.03
"""Leftovers shrink by this share per week, so the trends chart has a story."""
RISING_DISH = "Rice Pilaf"
"""One dish whose leftovers grow instead, so a dish row can show trend "up"."""


def _region(rng: random.Random, dish: KnownDish, factor: float, served_ml: float) -> RegionIn:
    _, typical = PROFILE.get(dish.name, (1.0, 22.0))
    pct = min(180.0, typical * factor * rng.lognormvariate(0.0, 0.45))
    volume = min(REFERENCE_ML[dish.category] * pct / 100.0, served_ml / 1.5)
    pct = volume / REFERENCE_ML[dish.category] * 100.0
    est = estimate_grams(volume, dish.category)
    unknown = rng.random() < UNKNOWN_SHARE
    return RegionIn(
        category=dish.category,
        dish_name=None if unknown else dish.name,
        match_score=None,
        volume_ml=round(volume, 1),
        grams=round(est.grams, 1),
        # Demo-only ±5% assumption; does not represent validated model accuracy.
        grams_low=round(round(est.grams, 1) * 0.95, 4),
        grams_high=round(round(est.grams, 1) * 1.05, 4),
        percent_left=None if unknown else round(pct, 1),
    )


def generate_demo(
    start: date, end: date, seed: int = DEFAULT_SEED
) -> tuple[list[PlateIn], list[ServingRecord]]:
    """Simulated plates for every day from ``start`` to ``end`` inclusive (hall-local times)."""
    plates: list[PlateIn] = []
    serving_totals: dict[tuple[date, str, str], ServingRecord] = {}
    day = start
    while day <= end:
        weeks_in = (day - start).days / 7.0
        for meal in WEEKEND_MEALS if day.weekday() >= 5 else WEEKDAY_MEALS:
            rng = random.Random(f"{seed}:{day.isoformat()}:{meal.name}")
            served_rng = random.Random(f"servings:{seed}:{day.isoformat()}:{meal.name}")
            weights = [PROFILE.get(d.name, (1.0, 22.0))[0] for d in meal.dishes]
            opens = datetime.combine(day, meal.opens)
            minutes = int((datetime.combine(day, meal.closes) - opens).total_seconds() // 60)
            for _ in range(round(meal.plates * rng.uniform(0.85, 1.15))):
                count = rng.choices((0, 1, 2, 3), weights=(12, 35, 35, 18))[0]
                chosen: list[KnownDish] = []
                served_count = count or served_rng.randint(1, 3)
                while len(chosen) < served_count:
                    pick = rng.choices(meal.dishes, weights=weights)[0]
                    if pick not in chosen:
                        chosen.append(pick)
                regions = []
                for dish in chosen:
                    drift = 1.0 + (0.05 if dish.name == RISING_DISH else -WEEKLY_IMPROVEMENT)
                    # Choose initial portions before leftovers, including clean plates.
                    served_ml = REFERENCE_ML[dish.category] * served_rng.uniform(0.85, 1.4)
                    key = (day, meal.name, dish.name)
                    if key not in serving_totals:
                        serving_totals[key] = ServingRecord(
                            hall="",
                            day=day,
                            meal=meal.name,
                            dish_name=dish.name,
                            servings=0,
                            served_kg=0,
                        )
                    record = serving_totals[key]
                    record.servings += 1
                    record.served_kg += estimate_grams(served_ml, dish.category).grams / 1000
                    if count:
                        regions.append(_region(rng, dish, max(0.3, drift**weeks_in), served_ml))
                plates.append(
                    PlateIn(
                        plate_id=f"{rng.getrandbits(64):016x}",
                        meal=meal.name,
                        observed_at=opens + timedelta(minutes=rng.randrange(minutes)),
                        regions=regions,
                    )
                )
        day += timedelta(days=1)
    return plates, list(serving_totals.values())


def generate_plates(start: date, end: date, seed: int = DEFAULT_SEED) -> list[PlateIn]:
    """Compatibility wrapper; plates and servings share one deterministic simulation."""
    return generate_demo(start, end, seed)[0]


def clear_seeded(db: Database, hall: str) -> None:
    """Delete seeded plates for ``hall``; live plates are never touched."""
    with Session(db.engine) as session:
        for table in (PlateEvent, Plate, ServingRecord):
            session.exec(
                delete(table).where(col(table.hall) == hall, col(table.data_source) == "seeded")
            )
        session.commit()


def seed(
    db: Database,
    *,
    hall: str,
    timezone: str,
    start: date,
    end: date,
    seed_value: int = DEFAULT_SEED,
) -> int:
    """Replace the hall's seeded plates with freshly generated ones; returns the plate count."""
    plates, servings = generate_demo(start, end, seed_value)
    clear_seeded(db, hall)
    with Session(db.engine) as session:
        for record in servings:
            record.hall = hall
            session.add(record)
        for plate in plates:
            add_plate(session, plate, hall=hall, timezone=timezone, data_source="seeded")
        session.commit()
    return len(plates)


def main(argv: Sequence[str] | None = None) -> None:
    cfg = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="weeks of data (4)")
    parser.add_argument("--end", type=date.fromisoformat, help="last day (default: today)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--database-url", default=cfg.database_url)
    parser.add_argument("--clear", action="store_true", help="only delete seeded plates")
    args = parser.parse_args(argv)

    db = Database(args.database_url)
    if args.clear:
        clear_seeded(db, cfg.hall)
        print(f"removed seeded plates for {cfg.hall} from {args.database_url}")
        return
    end: date = args.end or datetime.now(ZoneInfo(cfg.timezone)).date()
    start = end - timedelta(weeks=args.weeks) + timedelta(days=1)
    n = seed(db, hall=cfg.hall, timezone=cfg.timezone, start=start, end=end, seed_value=args.seed)
    print(f"seeded {n} plates for {cfg.hall}, {start} to {end}, into {args.database_url}")


if __name__ == "__main__":
    main()
