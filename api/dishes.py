"""Known USC Village dishes: station and food category, for labelling dashboard rows.

Taken from the Village menu of Saturday 14 Nov 2026 (``tests/fixtures``), as
cleaned and categorized by ``menu/``. Live plates carry the category the rig
saw; this table only fills in the station, which plates do not carry. A dish
not listed here simply gets ``station: null``.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import FoodCategory as C


@dataclass(frozen=True, slots=True)
class KnownDish:
    name: str
    station: str
    category: C


BRUNCH: tuple[KnownDish, ...] = (
    KnownDish("Boca Burger on White Bun", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("Beef Burger on Martin's Potato Bun", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("All Beef Hot Dog on White Bun", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("Assorted French Fries", "Grill", C.POTATOES_FRIED_ROASTED),
    KnownDish("Deli Sandwich", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("Assorted Bagels", "Sweets and Treats", C.BREAD),
    KnownDish("Scrambled Eggs", "Expo", C.EGGS),
    KnownDish("Boiled Eggs", "Expo", C.EGGS),
    KnownDish("Crispy Fried Tater Tots", "Expo", C.POTATOES_FRIED_ROASTED),
    KnownDish("Bacon", "Expo", C.RED_MEAT),
    KnownDish("Chicken Sausage Link", "Expo", C.POULTRY),
    KnownDish("Fried BBQ Tofu", "Expo", C.TOFU_PLANT_PROTEIN),
    KnownDish("Watermelon Glazed Grilled Chicken", "Expo", C.POULTRY),
    KnownDish("Roasted Broccoli", "Expo", C.COOKED_VEGETABLES),
    KnownDish("Chipotle Roasted Yams", "Expo", C.POTATOES_FRIED_ROASTED),
    KnownDish("Plain Pasta", "Expo", C.PASTA_PLAIN),
    KnownDish("Oatmeal", "Salad Bar", C.MASHED_PUREE),
    KnownDish("Assorted Breakfast Flaky Pastries", "Salad Bar", C.BAKED_DESSERT),
    KnownDish("Assorted Muffins", "Salad Bar", C.BAKED_DESSERT),
    KnownDish("Assorted Fruits", "Salad Bar", C.CUT_FRUIT),
    KnownDish("Plain Greek Yogurt", "Salad Bar", C.YOGURT_PUDDING),
    KnownDish("Made-to-Order Omelette", "Crepes", C.EGGS),
)

DINNER: tuple[KnownDish, ...] = (
    KnownDish("Boca Burger on White Bun", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("Beef Burger on Martin's Potato Bun", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("All Beef Hot Dog on White Bun", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("Assorted French Fries", "Grill", C.POTATOES_FRIED_ROASTED),
    KnownDish("Tuna Salad", "Grill", C.FISH_SEAFOOD),
    KnownDish("Deli Sandwich", "Grill", C.PIZZA_BURGER_SANDWICH),
    KnownDish("Sliced French Bread", "Grill", C.BREAD),
    KnownDish("Assorted Fruits", "Salad Bar", C.CUT_FRUIT),
    KnownDish("Salad Bar Salad", "Salad Bar", C.LEAFY_SALAD),
    KnownDish("Molly's Garden Vegetable Soup with Kidney Beans", "Salad Bar", C.SOUP_STEW),
    KnownDish("Chicken Noodle Soup with Egg Noodles", "Salad Bar", C.SOUP_STEW),
    KnownDish('Portobello Red Wine "Pot Roast"', "Flexitarian", C.COOKED_VEGETABLES),
    KnownDish("Garlic and Herb Roasted Chicken", "Flexitarian", C.POULTRY),
    KnownDish("Braised Beef Short Ribs", "Flexitarian", C.RED_MEAT),
    KnownDish("Braised Lentils with Garlic and Lemon", "Flexitarian", C.BEANS_LENTILS),
    KnownDish("Roasted Lemon Zest Asparagus", "Flexitarian", C.COOKED_VEGETABLES),
    KnownDish("Maple Roasted Carrots", "Flexitarian", C.COOKED_VEGETABLES),
    KnownDish("Rice Pilaf", "Flexitarian", C.RICE_GRAINS),
    KnownDish("Baked Potatoes", "Flexitarian", C.POTATOES_FRIED_ROASTED),
    KnownDish("Beef Chili with Beans", "Flexitarian", C.SOUP_STEW),
    KnownDish("French Style Crepe", "Sweets and Treats", C.BAKED_DESSERT),
)

CATALOG: dict[str, KnownDish] = {d.name: d for d in (*BRUNCH, *DINNER)}


def station_of(dish_name: str) -> str | None:
    known = CATALOG.get(dish_name)
    return known.station if known else None
