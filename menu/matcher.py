"""Match a segmented food region to one of today's menu dishes, or to "unknown".

The segmentation model only knows 21 coarse categories (``core.types.FoodCategory``);
the dashboard wants dish names. The menu narrows the problem: a region labelled
``poultry`` at dinner can only be one of the poultry dishes served at that meal.

Decision for one region:

1. **Candidates** are the meal's dishes (``MealMenu.candidates()``) whose category,
   as assigned by ``menu.categorizer``, equals the region's predicted category. The
   matcher can never return a dish that is not on the menu it was built with.
2. **No candidate**: unknown (the region goes to the review queue).
3. **One candidate**: that dish. Nothing was compared, so the match is only as good
   as the segmentation category; the score says so (see below).
4. **Several candidates**: the region is cropped to its mask's bounding box, pixels
   outside the mask are greyed out, and the crop is embedded with SigLIP. Each
   candidate's text embedding (an average over a few prompt templates) is compared
   with the crop; the scaled cosine similarities go through a softmax over the
   candidates. The top dish is returned if its probability is at least
   ``min_probability`` and beats the runner-up by at least ``min_margin``;
   otherwise the region is unknown.

Softmax rather than SigLIP's own per-pair sigmoid: the candidates are mutually
exclusive and the region is known (up to segmentation error) to be one of them, so
what matters is how the candidates compare with each other. SigLIP's sigmoid
probabilities for food crops against short dish names are typically all small,
which would make an absolute threshold on them meaningless. The learned bias cancels
in the softmax; the learned logit scale is kept so the softmax temperature is the
one SigLIP was trained with.

``DishMatch.score`` is ``region.confidence * p``, where ``p`` is the chosen dish's
probability among the candidates (1.0 when it was the only candidate). For an
unknown that came from SigLIP it is what the rejected top dish would have scored;
for an unknown without candidates it is 0. It is a ranking signal for the review
queue, not a calibrated probability.

The default thresholds are starting points, not tuned values: tune them with
``python -m eval.matching`` on labelled crops and record the run in ``eval/results``.

The SigLIP code path (``SiglipEmbedder``) needs the ``match`` extra
(transformers, torch, pillow) and network access to Hugging Face on first use. It
is written against the documented transformers API but has not been run in this
repository's CI, which cannot download the weights; every other part of this module
is unit-tested with a fake embedder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from core.types import DishMatch, FoodCategory, FoodRegion, ImageBGR, Mask
from menu.models import DailyMenu, MenuItem

Embedding = NDArray[np.float32]
ImageRGB = NDArray[np.uint8]

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
"""SigLIP 2, ViT-B/16 at 224 px: the smallest SigLIP 2 checkpoint, fast on CPU."""
DEFAULT_PROMPTS: tuple[str, ...] = (
    "this is a photo of {}.",
    "a photo of {} on a plate.",
    "a close-up photo of leftover {}.",
)
"""Prompt templates; ``{}`` is the lowercased dish name. Embeddings are averaged."""
DEFAULT_MIN_PROBABILITY = 0.5
DEFAULT_MIN_MARGIN = 0.15
DEFAULT_CROP_PADDING = 0.1
GREY = 128
"""Fill value for pixels outside the mask: SigLIP's processor maps 128 to roughly 0."""


class Embedder(Protocol):
    """An image-text model with a shared embedding space (SigLIP, or a fake in tests)."""

    @property
    def logit_scale(self) -> float:
        """Multiplier applied to cosine similarities before the softmax."""
        ...

    def embed_image(self, image: ImageRGB) -> Embedding:
        """L2-normalised embedding of one H x W x 3 RGB image, shape (D,)."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> Embedding:
        """L2-normalised embeddings of ``texts``, shape (len(texts), D)."""
        ...


class MatchReason(StrEnum):
    """Why the matcher answered the way it did."""

    MATCHED = "matched"
    """Several candidates; SigLIP picked one with enough probability and margin."""
    SINGLE_CANDIDATE = "single_candidate"
    """Only one dish on the menu has the region's category."""
    NO_CANDIDATES = "no_candidates"
    """No dish on the menu has the region's category."""
    LOW_PROBABILITY = "low_probability"
    """The top dish's probability is below ``min_probability``."""
    LOW_MARGIN = "low_margin"
    """The top two dishes are closer than ``min_margin``."""
    EMPTY_MASK = "empty_mask"
    """The region's mask has no pixels."""


@dataclass(frozen=True, slots=True)
class MatchDetail:
    """A ``DishMatch`` plus what it was decided from (for the review queue and eval)."""

    match: DishMatch
    reason: MatchReason
    candidates: tuple[str, ...]
    """Candidate dish names, in descending probability when SigLIP was run."""
    probabilities: tuple[float, ...]
    """Softmax probability per candidate (same order); empty if SigLIP was not run."""


def mask_crop(
    image: ImageBGR, mask: Mask, *, padding: float = DEFAULT_CROP_PADDING, fill: int = GREY
) -> ImageRGB | None:
    """Crop ``image`` to ``mask``'s bounding box and grey out the pixels outside the mask.

    Args:
        image: H x W x 3 BGR frame.
        mask: H x W boolean mask of the region.
        padding: margin added on each side, as a fraction of the box's width/height.
        fill: grey level for pixels outside the mask.

    Returns:
        The RGB crop, or None if the mask is empty.

    Raises:
        ValueError: image and mask sizes differ.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image {image.shape[:2]} and mask {mask.shape} differ in size")
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    pad_y, pad_x = round((y1 - y0) * padding), round((x1 - x0) * padding)
    h, w = mask.shape
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    crop = image[y0:y1, x0:x1, ::-1].copy()  # BGR -> RGB
    crop[~mask[y0:y1, x0:x1]] = fill
    return crop


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax of a 1-D array."""
    shifted = np.exp(logits - logits.max())
    result: NDArray[np.float64] = shifted / shifted.sum()
    return result


def menu_candidates(menu: DailyMenu, meal: str | None = None) -> list[MenuItem]:
    """Dishes a region may be matched to: one meal's candidates, or the whole day's.

    With ``meal=None`` the candidates of every meal are merged, deduplicated by dish
    key (first occurrence wins).

    Raises:
        KeyError: ``meal`` is not served on ``menu``'s day.
    """
    if meal is not None:
        found = menu.meal(meal)
        if found is None:
            served = ", ".join(m.name for m in menu.meals) or "none"
            raise KeyError(f"{meal!r} is not served on {menu.day} (served: {served})")
        return found.candidates()
    seen: dict[str, MenuItem] = {}
    for m in menu.meals:
        for item in m.candidates():
            seen.setdefault(item.key, item)
    return list(seen.values())


class MenuMatcher:
    """Matches regions to the dishes of one menu (one day, optionally one meal).

    Build one per menu and meal; text embeddings are computed once per dish and
    cached on the instance.
    """

    def __init__(
        self,
        menu: DailyMenu,
        embedder: Embedder,
        *,
        meal: str | None = None,
        prompts: Sequence[str] = DEFAULT_PROMPTS,
        min_probability: float = DEFAULT_MIN_PROBABILITY,
        min_margin: float = DEFAULT_MIN_MARGIN,
        crop_padding: float = DEFAULT_CROP_PADDING,
    ) -> None:
        """
        Args:
            menu: today's categorized menu (``menu.load_daily_menu``).
            embedder: image-text model, e.g. ``SiglipEmbedder()``.
            meal: restrict candidates to this meal; None uses every meal of the day.
            prompts: templates with one ``{}`` for the dish name; embeddings are averaged.
            min_probability: smallest softmax probability accepted for the top dish
                (default 0.5, untuned).
            min_margin: smallest probability gap between the top two dishes
                (default 0.15, untuned).
            crop_padding: margin around the mask's bounding box, as a fraction.

        Raises:
            KeyError: ``meal`` is not served that day.
            ValueError: no prompts, a prompt without ``{}``, or thresholds out of range.
        """
        if not prompts or any("{}" not in p for p in prompts):
            raise ValueError("prompts must be non-empty and each must contain '{}'")
        if not 0.0 <= min_probability <= 1.0 or not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_probability and min_margin must be in [0, 1]")
        self.menu = menu
        self.meal = meal
        self.embedder = embedder
        self.prompts = tuple(prompts)
        self.min_probability = min_probability
        self.min_margin = min_margin
        self.crop_padding = crop_padding
        self._by_category: dict[FoodCategory, list[MenuItem]] = {}
        for item in menu_candidates(menu, meal):
            if item.category is not None:
                self._by_category.setdefault(item.category, []).append(item)
        self._text_cache: dict[str, Embedding] = {}

    def candidates(self, category: FoodCategory) -> list[MenuItem]:
        """Menu dishes with ``category`` (uncategorized dishes are never candidates)."""
        return list(self._by_category.get(category, []))

    def text_embedding(self, item: MenuItem) -> Embedding:
        """Unit-length mean of the prompt embeddings for ``item``, cached by dish key."""
        cached = self._text_cache.get(item.key)
        if cached is not None:
            return cached
        name = item.name.lower()
        vectors = self.embedder.embed_texts([p.format(name) for p in self.prompts])
        mean = np.asarray(vectors, dtype=np.float32).mean(axis=0)
        embedding: Embedding = (mean / np.linalg.norm(mean)).astype(np.float32)
        self._text_cache[item.key] = embedding
        return embedding

    def rank(self, crop: ImageRGB, items: Sequence[MenuItem]) -> list[tuple[MenuItem, float]]:
        """Candidates with their softmax probability for ``crop``, best first."""
        image = np.asarray(self.embedder.embed_image(crop), dtype=np.float64)
        image = image / np.linalg.norm(image)
        texts = np.stack([self.text_embedding(i) for i in items]).astype(np.float64)
        probs = softmax(self.embedder.logit_scale * (texts @ image))
        order = np.argsort(-probs, kind="stable")
        return [(items[int(k)], float(probs[k])) for k in order]

    def explain(self, image: ImageBGR, region: FoodRegion) -> MatchDetail:
        """Match ``region`` of ``image`` and say why (see the module docstring)."""
        items = self.candidates(region.category)
        names = tuple(i.name for i in items)
        if not items:
            return MatchDetail(DishMatch(None, 0.0), MatchReason.NO_CANDIDATES, (), ())
        if len(items) == 1:
            match = DishMatch(items[0].name, float(region.confidence))
            return MatchDetail(match, MatchReason.SINGLE_CANDIDATE, names, ())
        crop = mask_crop(image, region.mask, padding=self.crop_padding)
        if crop is None:
            return MatchDetail(DishMatch(None, 0.0), MatchReason.EMPTY_MASK, names, ())
        ranked = self.rank(crop, items)
        ranked_names = tuple(i.name for i, _ in ranked)
        probs = tuple(p for _, p in ranked)
        top, p_top = ranked[0]
        if p_top < self.min_probability:
            reason = MatchReason.LOW_PROBABILITY
        elif p_top - probs[1] < self.min_margin:
            reason = MatchReason.LOW_MARGIN
        else:
            match = DishMatch(top.name, float(region.confidence) * p_top)
            return MatchDetail(match, MatchReason.MATCHED, ranked_names, probs)
        unknown = DishMatch(None, float(region.confidence) * p_top)
        return MatchDetail(unknown, reason, ranked_names, probs)

    def match(self, image: ImageBGR, region: FoodRegion) -> DishMatch:
        """The dish ``region`` shows, or ``DishMatch(None, ...)`` for unknown."""
        return self.explain(image, region).match


class SiglipEmbedder:
    """SigLIP / SigLIP 2 through Hugging Face transformers, loaded on first use.

    UNTESTED here: CI cannot reach Hugging Face. Written against the transformers
    docs for SigLIP 2 (``AutoModel``/``AutoProcessor``, ``get_image_features``,
    ``get_text_features``, text padded to 64 tokens as in training).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = AutoModel.from_pretrained(self.model_id).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self.device = device
        return self._model, self._processor

    @staticmethod
    def _to_numpy(features: Any) -> Embedding:
        # Recent transformers versions may return an output object instead of a tensor.
        tensor = getattr(features, "pooler_output", features)
        tensor = tensor / tensor.norm(p=2, dim=-1, keepdim=True)
        result: Embedding = tensor.float().cpu().numpy().astype(np.float32)
        return result

    @property
    def logit_scale(self) -> float:
        model, _ = self._load()
        return float(model.logit_scale.exp().item())

    def embed_image(self, image: ImageRGB) -> Embedding:
        import torch
        from PIL import Image

        model, processor = self._load()
        inputs = processor(images=Image.fromarray(image), return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        first: Embedding = self._to_numpy(features)[0]
        return first

    def embed_texts(self, texts: Sequence[str]) -> Embedding:
        import torch

        model, processor = self._load()
        inputs = processor(
            text=list(texts),
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        return self._to_numpy(features)
