from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest
from numpy.typing import NDArray

from core.types import DishMatch, FoodCategory, FoodRegion
from menu.cleaner import dish_key
from menu.matcher import (
    GREY,
    MatchReason,
    MenuMatcher,
    SiglipEmbedder,
    mask_crop,
    menu_candidates,
    softmax,
)
from menu.models import DailyMenu, ItemRole, MealMenu, MenuItem

C = FoodCategory


def item(name: str, category: FoodCategory | None, role: ItemRole = ItemRole.DISH) -> MenuItem:
    return MenuItem(name=name, key=dish_key(name), role=role, stations=["Main"], category=category)


def make_menu() -> DailyMenu:
    dinner = MealMenu(
        name="Dinner",
        items=[
            item("Garlic and Herb Roasted Chicken", C.POULTRY),
            item("Watermelon Glazed Grilled Chicken", C.POULTRY),
            item("Rice Pilaf", C.RICE_GRAINS),
            item("Mystery Special", None),
            item("Ranch Dressing", None, ItemRole.CONDIMENT),
        ],
    )
    lunch = MealMenu(
        name="Lunch",
        items=[item("Rice Pilaf", C.RICE_GRAINS), item("Plain Pasta", C.PASTA_PLAIN)],
    )
    return DailyMenu(
        day=date(2026, 11, 14), hall="university-village", source="manual", meals=[dinner, lunch]
    )


class FakeEmbedder:
    """Axis 0 = roasted, axis 1 = glazed; the image vector is set per test."""

    def __init__(self, image_vector: Sequence[float], scale: float = 10.0) -> None:
        self.image_vector = np.asarray(image_vector, dtype=np.float32)
        self.scale = scale
        self.text_calls: list[list[str]] = []
        self.image_shapes: list[tuple[int, ...]] = []

    @property
    def logit_scale(self) -> float:
        return self.scale

    def embed_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        self.image_shapes.append(image.shape)
        return self.image_vector / np.linalg.norm(self.image_vector)

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.text_calls.append(list(texts))
        rows = []
        for t in texts:
            if "roasted" in t:
                rows.append([1.0, 0.0])
            elif "glazed" in t:
                rows.append([0.0, 1.0])
            else:
                rows.append([0.7071, 0.7071])
        return np.asarray(rows, dtype=np.float32)


def frame_and_region(
    category: FoodCategory, confidence: float = 0.9
) -> tuple[NDArray[np.uint8], FoodRegion]:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:20, 20:40] = True
    return image, FoodRegion(mask=mask, category=category, confidence=confidence)


def test_mask_crop_bounds_greys_and_converts_to_rgb() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 3:7] = True
    mask[2, 3] = False
    crop = mask_crop(image, mask, padding=0.0)
    assert crop is not None
    assert crop.shape == (4, 4, 3)
    assert tuple(crop[1, 1]) == (0, 0, 255)  # now RGB
    assert tuple(crop[0, 0]) == (GREY, GREY, GREY)


def test_mask_crop_padding_is_clipped_to_image() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:4, 0:4] = True
    crop = mask_crop(image, mask, padding=0.5)
    assert crop is not None and crop.shape == (6, 6, 3)


def test_mask_crop_empty_and_mismatched() -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    assert mask_crop(image, np.zeros((5, 5), dtype=bool)) is None
    with pytest.raises(ValueError):
        mask_crop(image, np.zeros((4, 5), dtype=bool))


def test_softmax_sums_to_one_and_is_stable() -> None:
    probs = softmax(np.array([1000.0, 1000.0, 0.0]))
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.5)


def test_menu_candidates_per_meal_and_whole_day() -> None:
    menu = make_menu()
    dinner = [i.name for i in menu_candidates(menu, "dinner")]
    assert "Ranch Dressing" not in dinner and len(dinner) == 4
    day = [i.name for i in menu_candidates(menu)]
    assert day.count("Rice Pilaf") == 1 and "Plain Pasta" in day
    with pytest.raises(KeyError):
        menu_candidates(menu, "Breakfast")


def test_candidates_skip_uncategorized_and_other_categories() -> None:
    matcher = MenuMatcher(make_menu(), FakeEmbedder([1, 0]), meal="Dinner")
    assert [i.name for i in matcher.candidates(C.RICE_GRAINS)] == ["Rice Pilaf"]
    assert len(matcher.candidates(C.POULTRY)) == 2
    assert matcher.candidates(C.PASTA_PLAIN) == []


def test_no_candidates_is_unknown() -> None:
    matcher = MenuMatcher(make_menu(), FakeEmbedder([1, 0]), meal="Dinner")
    detail = matcher.explain(*frame_and_region(C.ICE_CREAM))
    assert detail.match == DishMatch(None, 0.0)
    assert detail.reason is MatchReason.NO_CANDIDATES


def test_single_candidate_skips_siglip() -> None:
    embedder = FakeEmbedder([1, 0])
    matcher = MenuMatcher(make_menu(), embedder, meal="Dinner")
    detail = matcher.explain(*frame_and_region(C.RICE_GRAINS, confidence=0.8))
    assert detail.match == DishMatch("Rice Pilaf", 0.8)
    assert detail.reason is MatchReason.SINGLE_CANDIDATE
    assert embedder.image_shapes == [] and embedder.text_calls == []


def test_siglip_picks_the_closer_dish() -> None:
    embedder = FakeEmbedder([1, 0.1])
    matcher = MenuMatcher(make_menu(), embedder, meal="Dinner")
    image, region = frame_and_region(C.POULTRY, confidence=0.5)
    detail = matcher.explain(image, region)
    assert detail.reason is MatchReason.MATCHED
    assert detail.match.dish_name == "Garlic and Herb Roasted Chicken"
    assert detail.candidates[0] == "Garlic and Herb Roasted Chicken"
    assert sum(detail.probabilities) == pytest.approx(1.0)
    assert detail.match.score == pytest.approx(0.5 * detail.probabilities[0])
    assert embedder.image_shapes == [(12, 24, 3)]  # 10x20 mask box plus 10% padding
    assert matcher.match(image, region) == detail.match


def test_prompts_are_lowercased_and_text_embeddings_cached() -> None:
    embedder = FakeEmbedder([0, 1])
    matcher = MenuMatcher(make_menu(), embedder, meal="Dinner", prompts=("a photo of {}.",))
    image, region = frame_and_region(C.POULTRY)
    first = matcher.explain(image, region)
    matcher.explain(image, region)
    assert first.match.dish_name == "Watermelon Glazed Grilled Chicken"
    assert len(embedder.text_calls) == 2  # one per dish, not per call
    assert embedder.text_calls[0] == ["a photo of garlic and herb roasted chicken."]


def test_text_embedding_is_unit_mean_of_prompts() -> None:
    matcher = MenuMatcher(make_menu(), FakeEmbedder([1, 0]))
    vec = matcher.text_embedding(item("Garlic and Herb Roasted Chicken", C.POULTRY))
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    np.testing.assert_allclose(vec, [1.0, 0.0], atol=1e-6)


def test_rank_orders_by_probability() -> None:
    menu = make_menu()
    matcher = MenuMatcher(menu, FakeEmbedder([0.2, 1.0]), meal="Dinner")
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    ranked = matcher.rank(crop, matcher.candidates(C.POULTRY))
    assert ranked[0][0].name == "Watermelon Glazed Grilled Chicken"
    assert ranked[0][1] > ranked[1][1]


def test_low_probability_is_unknown() -> None:
    # Equal similarity: each of two candidates gets 0.5, below a 0.6 floor.
    matcher = MenuMatcher(make_menu(), FakeEmbedder([1, 1]), meal="Dinner", min_probability=0.6)
    detail = matcher.explain(*frame_and_region(C.POULTRY))
    assert detail.match.dish_name is None
    assert detail.reason is MatchReason.LOW_PROBABILITY
    assert len(detail.probabilities) == 2


def test_low_margin_is_unknown() -> None:
    matcher = MenuMatcher(
        make_menu(),
        FakeEmbedder([1, 0.9], scale=1.0),
        meal="Dinner",
        min_probability=0.0,
        min_margin=0.3,
    )
    detail = matcher.explain(*frame_and_region(C.POULTRY))
    assert detail.match.dish_name is None
    assert detail.reason is MatchReason.LOW_MARGIN


def test_empty_mask_is_unknown() -> None:
    matcher = MenuMatcher(make_menu(), FakeEmbedder([1, 0]), meal="Dinner")
    image, region = frame_and_region(C.POULTRY)
    empty = FoodRegion(mask=np.zeros_like(region.mask), category=C.POULTRY, confidence=0.9)
    assert matcher.explain(image, empty).reason is MatchReason.EMPTY_MASK


def test_never_returns_a_dish_off_the_menu() -> None:
    menu = make_menu()
    names = {i.name for i in menu_candidates(menu)}
    for vector in ([1, 0], [0, 1], [1, 1], [-1, 0.2]):
        matcher = MenuMatcher(menu, FakeEmbedder(vector), min_probability=0.0, min_margin=0.0)
        for category in FoodCategory:
            got = matcher.match(*frame_and_region(category)).dish_name
            assert got is None or got in names


@pytest.mark.parametrize(
    "kwargs",
    [{"prompts": ()}, {"prompts": ("no slot",)}, {"min_probability": 1.5}, {"min_margin": -0.1}],
)
def test_invalid_settings_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MenuMatcher(make_menu(), FakeEmbedder([1, 0]), **kwargs)  # type: ignore[arg-type]


def test_siglip_embedder_is_lazy() -> None:
    embedder = SiglipEmbedder("some/model", device="cpu")
    assert embedder._model is None and embedder.model_id == "some/model"


def test_siglip_to_numpy_normalises_tensors_and_output_objects() -> None:
    torch = pytest.importorskip("torch")
    raw = torch.tensor([[3.0, 4.0]])
    np.testing.assert_allclose(SiglipEmbedder._to_numpy(raw), [[0.6, 0.8]], atol=1e-6)

    class Output:
        pooler_output = raw

    np.testing.assert_allclose(SiglipEmbedder._to_numpy(Output()), [[0.6, 0.8]], atol=1e-6)
