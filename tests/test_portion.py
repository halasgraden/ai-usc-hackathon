from __future__ import annotations

import pytest

from core.types import FoodCategory
from portion.grams import estimate_grams, load_densities, percent_left

# The density column of the spec's table, typed in independently of the YAML.
SPEC_TABLE = {
    "rice_grains": 0.73, "pasta_plain": 0.55, "pasta_saucy": 1.00, "soup_stew": 1.03,
    "mashed_puree": 1.0, "beans_lentils": 0.85, "poultry": 0.59, "red_meat": 0.6,
    "fish_seafood": 0.7, "eggs": 0.6, "tofu_plant_protein": 0.65, "raw_vegetables": 0.40,
    "cooked_vegetables": 0.6, "leafy_salad": 0.2, "potatoes_fried_roasted": 0.5, "bread": 0.25,
    "pizza_burger_sandwich": 0.45, "baked_dessert": 0.42, "ice_cream": 0.56,
    "yogurt_pudding": 1.03, "cut_fruit": 0.55,
}  # fmt: skip


def test_density_table_matches_the_spec() -> None:
    table = load_densities()
    assert {c.value: d.value for c, d in table.items()} == SPEC_TABLE
    assert set(table) == set(FoodCategory)


def test_ranges_bracket_the_value_and_estimates_are_wider() -> None:
    table = load_densities()
    for density in table.values():
        assert density.low < density.value < density.high
    published, estimate = table[FoodCategory.RICE_GRAINS], table[FoodCategory.LEAFY_SALAD]
    assert published.basis == "published" and estimate.basis == "estimate"
    assert (estimate.high - estimate.low) / estimate.value > (
        published.high - published.low
    ) / published.value


def test_grams_is_volume_times_density() -> None:
    portion = estimate_grams(200.0, FoodCategory.RICE_GRAINS)
    assert portion.grams == pytest.approx(146.0)
    assert portion.grams_low < portion.grams < portion.grams_high
    assert estimate_grams(0.0, FoodCategory.BREAD).grams == 0.0


def test_percent_left() -> None:
    assert percent_left(50.0, 200.0) == pytest.approx(25.0)
    assert percent_left(300.0, 200.0) == pytest.approx(150.0)  # larger than the reference


@pytest.mark.parametrize(("left", "served"), [(-1.0, 100.0), (10.0, 0.0), (10.0, -5.0)])
def test_percent_left_rejects_bad_input(left: float, served: float) -> None:
    with pytest.raises(ValueError):
        percent_left(left, served)


def test_negative_volume_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_grams(-1.0, FoodCategory.BREAD)
