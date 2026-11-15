from __future__ import annotations

import pytest

from menu.cleaner import CleaningRules, classify_role, clean_meal, dish_key, normalize_name
from menu.models import DailyMenu, ItemRole, MealMenu, RawMeal, RawMenuItem, RawStation


def role_of(rules: CleaningRules, name: str, *, tagged: bool = True) -> ItemRole:
    entry = RawMenuItem(item=name, preferences=["vegan"] if tagged else [])
    return classify_role(entry, rules)


def by_name(meal: MealMenu) -> dict[str, ItemRole]:
    return {item.name: item.role for item in meal.items}


def test_normalize_name_and_key() -> None:
    assert normalize_name("  Mac &amp; Cheese" + chr(160) + " Bake ") == "Mac & Cheese Bake"
    assert dish_key('Portobello Red Wine "Pot Roast"') == "portobello red wine pot roast"
    assert dish_key("Olive/Canola Oil Blend") == "olive canola oil blend"
    assert dish_key("Jalapeño Poppers") == "jalapeno poppers"


@pytest.mark.parametrize(
    ("name", "tagged"),
    [
        ("GRILL ITEMS AVAILABLE AT 11AM", False),
        ("SALAD BAR", False),
        ("Made to Order Omelettes Available Until 2pm", False),
        # A notice that carries diet tags and is not in capitals.
        ("Little Northern Bakehouse Allergen Free Bread Available Upon Request", True),
    ],
)
def test_notices(rules: CleaningRules, name: str, tagged: bool) -> None:
    assert role_of(rules, name, tagged=tagged) is ItemRole.NOTICE


@pytest.mark.parametrize(
    "name",
    [
        "Chipotle Lime Cream Sauce",
        "Cheddar Cheese Sauce",
        "Sunflower Seed Butter",
        "Butter Balls",
        "Plain Cream Cheese",
        "Grape Jelly",
        "Strawberry Preserves",
        "Spicy Salsa Verde",
        "Creamy Lime Cilantro Mustard Dressing",
        "Oregano, Dijon and Red Wine Vinaigrette",
        "Olive/Canola Oil Blend",
        "Balsamic Vinegar",
        "Traditional Hummus",
        "Whipped Topping",
        "Powdered Sugar",
    ],
)
def test_condiments(rules: CleaningRules, name: str) -> None:
    assert role_of(rules, name) is ItemRole.CONDIMENT


@pytest.mark.parametrize(
    "name",
    [
        "Butter Chicken",  # "butter" is not the head noun
        "Pasta with Marinara Sauce",  # ends in "sauce" but is a dish
        "Chicken in Black Bean Sauce",
        "Butternut Squash Soup",
        "Peanut Butter Cookie",
        "Sliced French Bread",  # "sliced" alone does not make a component
        "Scrambled Eggs",
        "Tuna Salad",
        "Braised Beef Short Ribs",  # untagged but not capitals: still a dish
        "BBQ",  # capitals but tagged
    ],
)
def test_dishes_are_not_filtered(rules: CleaningRules, name: str) -> None:
    assert role_of(rules, name) is ItemRole.DISH


def test_untagged_mixed_case_dish_is_kept(rules: CleaningRules) -> None:
    assert role_of(rules, "Sliced Roast Beef", tagged=False) is ItemRole.COMPONENT
    assert role_of(rules, "Braised Beef Short Ribs", tagged=False) is ItemRole.DISH


@pytest.mark.parametrize(
    "name",
    ["Chopped Bacon", "Crumbled Tofu", "Egg Whites", "Eggs", "Spinach", "Salami", "Pepitas",
     "Shredded Monterey Jack and Cheddar Cheese Blend", "Country Style Marinated Olives",
     "Oat n' Honey Granola", "Dried Cranberries", "Assorted Sliced Cheese"],
)  # fmt: skip
def test_components(rules: CleaningRules, name: str) -> None:
    assert role_of(rules, name) is ItemRole.COMPONENT


def test_unserved_meals_are_dropped(cleaned: DailyMenu) -> None:
    assert [m.name for m in cleaned.meals] == ["Brunch", "Dinner"]
    assert cleaned.meal("breakfast") is None
    assert cleaned.day.isoformat() == "2026-11-14"


def test_every_raw_entry_is_accounted_for(cleaned: DailyMenu) -> None:
    """72 brunch entries, 6 of them repeats across stations, plus 2 implied dishes."""
    brunch = cleaned.meal("Brunch")
    assert brunch is not None
    real = [i for i in brunch.items if not i.synthetic]
    assert len(real) == 66
    assert len({i.key for i in brunch.items}) == len(brunch.items)


def test_brunch_roles(cleaned: DailyMenu) -> None:
    brunch = cleaned.meal("Brunch")
    assert brunch is not None
    counts = {role: sum(i.role is role for i in brunch.items) for role in ItemRole}
    assert counts == {
        ItemRole.DISH: 28,
        ItemRole.COMPONENT: 27,
        ItemRole.CONDIMENT: 9,
        ItemRole.NOTICE: 4,
    }
    roles = by_name(brunch)
    assert roles["Oatmeal"] is ItemRole.DISH
    assert roles["LUNCH STYLE OPTIONS AVAILABLE AT 11AM"] is ItemRole.NOTICE


def test_repeats_merge_and_station_spelling_is_unified(cleaned: DailyMenu) -> None:
    brunch = cleaned.meal("Brunch")
    assert brunch is not None
    bread = next(i for i in brunch.items if i.name == "Sliced French Bread")
    # The API spells this station "Sweet and Treats" at brunch.
    assert bread.stations == ["Grill", "Sweets and Treats"]
    assert bread.allergens == ["gluten"]


def test_implied_dishes(cleaned: DailyMenu) -> None:
    brunch, dinner = cleaned.meal("Brunch"), cleaned.meal("Dinner")
    assert brunch is not None and dinner is not None
    synthetic_brunch = {i.name: i for i in brunch.items if i.synthetic}
    assert set(synthetic_brunch) == {"Made-to-Order Omelette", "Deli Sandwich"}
    assert synthetic_brunch["Made-to-Order Omelette"].stations == ["Crepes"]
    assert synthetic_brunch["Deli Sandwich"].stations == ["Grill"]
    # No omelette bar at dinner; the salad bar header implies a salad.
    assert {i.name for i in dinner.items if i.synthetic} == {"Deli Sandwich", "Salad Bar Salad"}


def test_dinner_candidates(cleaned: DailyMenu) -> None:
    dinner = cleaned.meal("Dinner")
    assert dinner is not None
    names = {i.name for i in dinner.candidates()}
    assert len(names) == 23
    assert "Garlic and Herb Roasted Chicken" in names
    assert 'Portobello Red Wine "Pot Roast"' in names
    assert "Ranch Dressing" not in names
    assert "Chocolate Chips" not in names
    assert "SALAD BAR" not in names


def test_composite_needs_enough_matches(rules: CleaningRules) -> None:
    meal = RawMeal(
        name="Lunch",
        stations=[RawStation(station="Grill", menu=[RawMenuItem(item="Sliced Turkey")])],
    )
    assert not any(i.synthetic for i in clean_meal(meal, rules).items)


def test_blank_entries_are_ignored(rules: CleaningRules) -> None:
    meal = RawMeal(
        name="Lunch",
        stations=[
            RawStation(station="Expo", menu=[RawMenuItem(item="  "), RawMenuItem(item="Rice")])
        ],
    )
    assert [i.name for i in clean_meal(meal, rules).items] == ["Rice"]
