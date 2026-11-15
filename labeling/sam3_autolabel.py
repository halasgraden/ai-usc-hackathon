"""Auto-label plate photos with SAM 3 (offline, on a GPU; never in the live path)."""

from __future__ import annotations

from collections.abc import Iterable

from core.types import SEG_CLASSES, FoodCategory, FoodRegion, ImageBGR

# Short noun phrases, the form SAM 3's concept prompts expect. Tune them
# against a handful of images before a full labeling run.
CATEGORY_PROMPTS: dict[FoodCategory, str] = {
    FoodCategory.RICE_GRAINS: "cooked rice",
    FoodCategory.PASTA_PLAIN: "plain pasta",
    FoodCategory.PASTA_SAUCY: "pasta in sauce",
    FoodCategory.SOUP_STEW: "soup",
    FoodCategory.MASHED_PUREE: "mashed potatoes",
    FoodCategory.BEANS_LENTILS: "cooked beans",
    FoodCategory.POULTRY: "chicken",
    FoodCategory.RED_MEAT: "meat",
    FoodCategory.FISH_SEAFOOD: "fish",
    FoodCategory.EGGS: "eggs",
    FoodCategory.TOFU_PLANT_PROTEIN: "tofu",
    FoodCategory.RAW_VEGETABLES: "raw vegetables",
    FoodCategory.COOKED_VEGETABLES: "cooked vegetables",
    FoodCategory.LEAFY_SALAD: "salad greens",
    FoodCategory.POTATOES_FRIED_ROASTED: "french fries",
    FoodCategory.BREAD: "bread",
    FoodCategory.PIZZA_BURGER_SANDWICH: "sandwich",
    FoodCategory.BAKED_DESSERT: "pastry",
    FoodCategory.ICE_CREAM: "ice cream",
    FoodCategory.YOGURT_PUDDING: "yogurt",
    FoodCategory.CUT_FRUIT: "cut fruit",
}


def prompts_for(categories: Iterable[FoodCategory]) -> dict[FoodCategory, str]:
    """Text prompts for the categories on today's menu, in ``FoodCategory`` order."""
    wanted = set(categories)
    return {c: CATEGORY_PROMPTS[c] for c in FoodCategory if c in wanted}


def class_index(category: FoodCategory) -> int:
    """Index of a food category in the segmentation model's class list."""
    return SEG_CLASSES.index(category.value)


def autolabel_image(image: ImageBGR, prompts: dict[FoodCategory, str]) -> list[FoodRegion]:
    """Run SAM 3 once per prompt and return every instance mask it finds."""
    raise NotImplementedError("hour 2-3, Colab GPU: needs gated SAM 3 weights")


def to_yolo_seg_lines(regions: list[FoodRegion]) -> list[str]:
    """Convert masks to Ultralytics segmentation label lines.

    One line per region: ``<class index> x1 y1 x2 y2 ...`` with polygon
    coordinates normalized to 0-1 by image width and height.
    """
    raise NotImplementedError("hour 2-3: cv2.findContours on each mask, largest contour")
