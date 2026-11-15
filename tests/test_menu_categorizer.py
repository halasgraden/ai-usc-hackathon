from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.types import FoodCategory as C
from menu.__main__ import main
from menu.categorizer import Categorizer, CategoryCache
from menu.models import DailyMenu, ItemRole, MenuItem

EXPECTED = {
    "Boca Burger on White Bun": C.PIZZA_BURGER_SANDWICH,
    "All Beef Hot Dog on White Bun": C.PIZZA_BURGER_SANDWICH,
    "Beef Burger on Martin's Potato Bun": C.PIZZA_BURGER_SANDWICH,
    "Assorted French Fries": C.POTATOES_FRIED_ROASTED,
    "Tuna Salad": C.FISH_SEAFOOD,
    "Sliced French Bread": C.BREAD,
    "Assorted Bagels": C.BREAD,
    "Scrambled Eggs": C.EGGS,
    "Crispy Fried Tater Tots": C.POTATOES_FRIED_ROASTED,
    "Bacon": C.RED_MEAT,
    "Chicken Sausage Link": C.POULTRY,
    "Fried BBQ Tofu": C.TOFU_PLANT_PROTEIN,
    "Watermelon Glazed Grilled Chicken": C.POULTRY,
    "Roasted Broccoli": C.COOKED_VEGETABLES,
    "Chipotle Roasted Yams": C.POTATOES_FRIED_ROASTED,
    "Plain Pasta": C.PASTA_PLAIN,
    "Oatmeal": C.MASHED_PUREE,
    "Assorted Breakfast Flaky Pastries": C.BAKED_DESSERT,
    "Assorted Muffins": C.BAKED_DESSERT,
    "Assorted Fruits": C.CUT_FRUIT,
    "Coconut Yogurt": C.YOGURT_PUDDING,
    "Made-to-Order Omelette": C.EGGS,
    "Deli Sandwich": C.PIZZA_BURGER_SANDWICH,
    "Molly's Garden Vegetable Soup with Kidney Beans": C.SOUP_STEW,
    "Chicken Noodle Soup with Egg Noodles": C.SOUP_STEW,
    # Vegan, so the "pot roast" keyword must not file it under red meat.
    'Portobello Red Wine "Pot Roast"': C.COOKED_VEGETABLES,
    "Garlic and Herb Roasted Chicken": C.POULTRY,
    "Braised Beef Short Ribs": C.RED_MEAT,
    "Braised Lentils with Garlic and Lemon": C.BEANS_LENTILS,
    "Roasted Lemon Zest Asparagus": C.COOKED_VEGETABLES,
    "Maple Roasted Carrots": C.COOKED_VEGETABLES,
    "Rice Pilaf": C.RICE_GRAINS,
    "Baked Potatoes": C.POTATOES_FRIED_ROASTED,
    "Beef Chili with Beans": C.SOUP_STEW,
    "French Style Crepe": C.BAKED_DESSERT,
    "Salad Bar Salad": C.LEAFY_SALAD,
}


def dish(
    name: str, *, stations: list[str] | None = None, prefs: list[str] | None = None
) -> MenuItem:
    from menu.cleaner import dish_key

    return MenuItem(
        name=name,
        key=dish_key(name),
        role=ItemRole.DISH,
        stations=stations or ["Expo"],
        preferences=prefs or [],
    )


def test_real_menu_categories(categorized: DailyMenu) -> None:
    got = {i.name: i.category for meal in categorized.meals for i in meal.candidates()}
    assert {name: got[name] for name in EXPECTED} == EXPECTED


def test_every_candidate_on_the_sample_day_is_categorized(categorized: DailyMenu) -> None:
    assert [i.name for meal in categorized.meals for i in meal.uncategorized()] == []


def test_condiments_and_notices_get_no_category(categorized: DailyMenu) -> None:
    for meal in categorized.meals:
        for item in meal.items:
            if item.role in (ItemRole.CONDIMENT, ItemRole.NOTICE):
                assert item.category is None


def test_apply_does_not_mutate_its_input(cleaned: DailyMenu) -> None:
    Categorizer.load().apply(cleaned)
    brunch = cleaned.meal("Brunch")
    assert brunch is not None
    assert next(i for i in brunch.items if i.name == "Oatmeal").category is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Green Beans Almondine", C.COOKED_VEGETABLES),
        ("Black Bean Soup", C.SOUP_STEW),
        ("Baked Mac and Cheese", C.PASTA_SAUCY),
        ("Orange Chicken", C.POULTRY),
        ("Hamburger Steak", C.RED_MEAT),  # "ham" must match as a whole word only
        ("Vanilla Soft Serve", C.ICE_CREAM),
        ("Frozen Yogurt", C.ICE_CREAM),
        ("Mystery Special", None),
    ],
)
def test_rule_order_and_word_boundaries(name: str, expected: C | None) -> None:
    category, source = Categorizer.load().categorize(dish(name))
    assert category == expected
    assert source == ("rule" if expected else None)


def test_station_restricted_rule() -> None:
    categorizer = Categorizer.load()
    assert categorizer.categorize(dish("Carrots", stations=["Salad Bar"]))[0] is C.RAW_VEGETABLES
    assert categorizer.categorize(dish("Carrots", stations=["Expo"]))[0] is C.COOKED_VEGETABLES


def test_diet_tags_block_animal_categories() -> None:
    categorizer = Categorizer.load()
    assert categorizer.categorize(dish("Vegan Chicken Nuggets", prefs=["vegan"]))[0] is None
    assert categorizer.categorize(dish("Chicken Nuggets"))[0] is C.POULTRY
    assert categorizer.categorize(dish("Egg Fried Rice", prefs=["vegetarian"]))[0] is C.EGGS
    assert categorizer.categorize(dish("Just Egg Fried Rice", prefs=["vegan"]))[0] is C.RICE_GRAINS


def test_override_wins_and_ignores_case_and_punctuation() -> None:
    base = Categorizer.load()
    categorizer = Categorizer(
        base.rules, overrides={"SHEPHERD'S pie": C.MASHED_PUREE, "Rice Pilaf": None}
    )
    assert categorizer.categorize(dish("Shepherd's Pie")) == (C.MASHED_PUREE, "override")
    assert categorizer.categorize(dish("Rice Pilaf")) == (None, "override")


def test_cache_records_rule_decisions_and_is_reused(tmp_path: Path) -> None:
    path = tmp_path / "category_cache.json"
    first = Categorizer.load(cache=CategoryCache(path))
    assert first.categorize(dish("Rice Pilaf")) == (C.RICE_GRAINS, "rule")
    assert first.cache is not None
    first.cache.save()
    assert json.loads(path.read_text())["entries"] == {"rice pilaf": "rice_grains"}

    second = Categorizer.load(cache=CategoryCache(path))
    assert second.categorize(dish("Rice Pilaf")) == (C.RICE_GRAINS, "cache")


def test_cache_is_discarded_when_rules_change(tmp_path: Path) -> None:
    path = tmp_path / "category_cache.json"
    path.write_text(json.dumps({"fingerprint": "old", "entries": {"rice pilaf": "bread"}}))
    categorizer = Categorizer.load(cache=CategoryCache(path))
    assert categorizer.categorize(dish("Rice Pilaf")) == (C.RICE_GRAINS, "rule")


def test_corrupt_cache_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "category_cache.json"
    path.write_text("{oops")
    assert Categorizer.load(cache=CategoryCache(path)).categorize(dish("Bacon"))[0] is C.RED_MEAT


def test_cli_renders_the_sample(capsys: pytest.CaptureFixture[str]) -> None:
    sample = Path(__file__).parent / "fixtures" / "village_2026-11-14.json"
    assert main(["--file", str(sample)]) == 0
    out = capsys.readouterr().out
    assert "Brunch: 28 plate candidates" in out
    assert "Dinner: 23 plate candidates" in out
    assert "UNCATEGORIZED" not in out


def test_cli_reports_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--file", str(tmp_path / "nope.json")]) == 1
    assert "error:" in capsys.readouterr().err
