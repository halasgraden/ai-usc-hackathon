"""Evaluate the menu matcher on labelled crops and write the numbers to a file.

This script is the only source of menu-matching numbers in the README: it writes
``eval/results/<name>.json`` and ``eval/results/<name>.md``. Nobody types a number by hand.

    python -m eval.matching --labels data/match_eval/labels.csv --name match-v1

The labels file is CSV or JSON (a list of objects) with these fields; paths are relative
to the labels file:

``crop``      image of one food region (any format Pillow reads)
``dish``      true dish name exactly as on the menu; empty or ``unknown`` for food that
              is not on the menu (the right answer is then "unknown")
``menu``      a saved menu JSON (a cached API response, or ``python -m menu --json`` output)
``date``      instead of ``menu``: ISO date, read from the cache
              ``<data-dir>/menus/<hall>/<date>.json`` (never fetched, so runs repeat)
``meal``      optional meal name; candidates then come from that meal only
``category``  optional ``FoodCategory``; defaults to the true dish's own category
``mask``      optional mask image (non-zero = region); defaults to the whole crop

Without a ``category`` column the matcher is given the correct category, so these
numbers measure the match stage alone, not segmentation plus matching; the report
says which. Accuracy counts an "unknown" answer as wrong; "accuracy when answered"
leaves unknowns out. Chance accuracy is the mean of 1 / (number of candidates):
what picking a candidate at random would score.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.types import FoodCategory, FoodRegion, ImageBGR, Mask
from menu.categorizer import Categorizer
from menu.cleaner import CleaningRules, clean_menu
from menu.importer import DEFAULT_HALL, cache_path, load_menu_file
from menu.matcher import (
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_PROBABILITY,
    DEFAULT_MODEL_ID,
    MatchDetail,
    MenuMatcher,
    SiglipEmbedder,
    menu_candidates,
)
from menu.models import DailyMenu

RESULTS_DIR = Path(__file__).parent / "results"
MIN_EXAMPLES = 5
UNKNOWN_LABELS = frozenset({"", "unknown"})


@dataclass(frozen=True, slots=True)
class Example:
    """One labelled crop."""

    crop: Path
    dish: str | None
    """True dish name; None means the food is not on the menu."""
    menu_ref: str
    """A menu file path or an ISO date (see the module docstring)."""
    meal: str | None = None
    category: FoodCategory | None = None
    mask: Path | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    true_dish: str | None
    predicted: str | None
    reason: str
    n_candidates: int
    score: float


@dataclass(frozen=True, slots=True)
class DishResult:
    name: str
    examples: int
    correct: int
    unknown: int
    accuracy: float
    unknown_rate: float
    accuracy_when_answered: float | None
    """None if every example was answered "unknown"."""

    @property
    def reliable(self) -> bool:
        return self.examples >= MIN_EXAMPLES


@dataclass(frozen=True, slots=True)
class Report:
    overall: DishResult
    dishes: list[DishResult]
    off_menu_examples: int
    off_menu_rejected: int
    """Off-menu examples the matcher correctly called unknown."""
    chance_accuracy: float | None
    """Mean over on-menu examples of 1/len(candidates) (0 with no candidates); None if no
    example had a candidate."""
    reasons: dict[str, int] = field(default_factory=dict)
    confusions: list[tuple[str, str, int]] = field(default_factory=list)
    """(true dish, wrongly predicted dish, count), most frequent first."""


class Matcher(Protocol):
    def explain(self, image: ImageBGR, region: FoodRegion) -> MatchDetail: ...


def _optional(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_examples(path: Path) -> list[Example]:
    """Read a CSV or JSON labels file; relative paths resolve against its folder.

    Raises:
        ValueError: a row lacks ``crop``, or has neither ``menu`` nor ``date``, or names
            an unknown category.
    """
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".json":
        rows = list(json.loads(text))
    else:
        rows = list(csv.DictReader(text.splitlines()))
    base = path.parent
    examples = []
    for n, row in enumerate(rows, start=1):
        crop = _optional(row, "crop")
        menu_file, day = _optional(row, "menu"), _optional(row, "date")
        if crop is None or (menu_file is None and day is None):
            raise ValueError(f"{path} row {n}: needs 'crop' and one of 'menu' or 'date'")
        dish = (_optional(row, "dish") or "").strip()
        category = _optional(row, "category")
        mask = _optional(row, "mask")
        try:
            parsed_category = FoodCategory(category) if category else None
        except ValueError as exc:
            raise ValueError(f"{path} row {n}: unknown category {category!r}") from exc
        examples.append(
            Example(
                crop=base / crop,
                dish=None if dish.casefold() in UNKNOWN_LABELS else dish,
                menu_ref=str(base / menu_file) if menu_file else str(day),
                meal=_optional(row, "meal"),
                category=parsed_category,
                mask=base / mask if mask else None,
            )
        )
    return examples


def load_menu(ref: str, *, data_dir: Path = Path("data"), hall: str = DEFAULT_HALL) -> DailyMenu:
    """A categorized menu from a file path or an ISO date (cache only, no network).

    Accepts a raw API response or a cleaned ``DailyMenu`` dump. Raises ``OSError`` or
    ``menu.importer.MenuError`` if it cannot be read.
    """
    try:
        day = date.fromisoformat(ref)
    except ValueError:
        path = Path(ref)
    else:
        path = cache_path(data_dir / "menus", hall, day)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "day" in payload and "hall" in payload:
        return Categorizer.load().apply(DailyMenu.model_validate(payload))
    raw = load_menu_file(path)
    return Categorizer.load().apply(clean_menu(raw, CleaningRules.load(), hall=hall, source="file"))


def true_category(example: Example, menu: DailyMenu) -> FoodCategory | None:
    """The category to give the matcher: the label's, else the true dish's on the menu.

    Raises:
        ValueError: the true dish is not a candidate on that menu and meal.
    """
    if example.category is not None:
        return example.category
    if example.dish is None:
        return None
    for item in menu_candidates(menu, example.meal):
        if item.name == example.dish:
            return item.category
    raise ValueError(f"{example.crop}: {example.dish!r} is not a dish on menu {example.menu_ref}")


def load_image(path: Path) -> ImageBGR:
    """Read an image as H x W x 3 BGR uint8."""
    from PIL import Image

    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def load_mask(path: Path | None, shape: tuple[int, int]) -> Mask:
    """Mask image (non-zero = region), or an all-True mask when ``path`` is None."""
    if path is None:
        return np.ones(shape, dtype=bool)
    from PIL import Image

    with Image.open(path) as img:
        mask = np.asarray(img.convert("L")) > 0
    if mask.shape != shape:
        raise ValueError(f"mask {path} is {mask.shape}, crop is {shape}")
    return mask


def predict(
    examples: Sequence[Example],
    menus: Callable[[str], DailyMenu],
    matcher_for: Callable[[DailyMenu, str | None], Matcher],
    image_loader: Callable[[Path], ImageBGR] = load_image,
    mask_loader: Callable[[Path | None, tuple[int, int]], Mask] = load_mask,
) -> list[Prediction]:
    """Run a matcher over every example. Menus and matchers are built once per (menu, meal).

    Off-menu examples without a ``category`` cannot be run (the matcher needs one) and
    raise ``ValueError``.
    """
    menu_cache: dict[str, DailyMenu] = {}
    matchers: dict[tuple[str, str | None], Matcher] = {}
    out: list[Prediction] = []
    for ex in examples:
        menu = menu_cache.get(ex.menu_ref)
        if menu is None:
            menu = menu_cache[ex.menu_ref] = menus(ex.menu_ref)
        category = true_category(ex, menu)
        if category is None:
            raise ValueError(f"{ex.crop}: off-menu or uncategorized example needs a 'category'")
        key = (ex.menu_ref, ex.meal)
        matcher = matchers.get(key)
        if matcher is None:
            matcher = matchers[key] = matcher_for(menu, ex.meal)
        image = image_loader(ex.crop)
        mask = mask_loader(ex.mask, (image.shape[0], image.shape[1]))
        detail = matcher.explain(image, FoodRegion(mask=mask, category=category, confidence=1.0))
        out.append(
            Prediction(
                true_dish=ex.dish,
                predicted=detail.match.dish_name,
                reason=str(detail.reason),
                n_candidates=len(detail.candidates),
                score=detail.match.score,
            )
        )
    return out


def _result(name: str, preds: Sequence[Prediction]) -> DishResult:
    n = len(preds)
    correct = sum(p.predicted == p.true_dish for p in preds)
    unknown = sum(p.predicted is None for p in preds)
    answered = n - unknown
    return DishResult(
        name=name,
        examples=n,
        correct=correct,
        unknown=unknown,
        accuracy=correct / n if n else 0.0,
        unknown_rate=unknown / n if n else 0.0,
        accuracy_when_answered=correct / answered if answered else None,
    )


def score(predictions: Sequence[Prediction]) -> Report:
    """Per-dish and overall metrics over on-menu examples, plus off-menu rejection.

    Raises:
        ValueError: no on-menu examples.
    """
    on_menu = [p for p in predictions if p.true_dish is not None]
    off_menu = [p for p in predictions if p.true_dish is None]
    if not on_menu:
        raise ValueError("no on-menu examples to score")
    by_dish: dict[str, list[Prediction]] = {}
    for p in on_menu:
        assert p.true_dish is not None
        by_dish.setdefault(p.true_dish, []).append(p)
    with_candidates = [p for p in on_menu if p.n_candidates > 0]
    chance = (
        sum(1 / p.n_candidates for p in with_candidates) / len(on_menu) if with_candidates else None
    )
    confusions = Counter(
        (p.true_dish, p.predicted)
        for p in on_menu
        if p.predicted is not None and p.predicted != p.true_dish
    )
    return Report(
        overall=_result("overall", on_menu),
        dishes=[_result(name, by_dish[name]) for name in sorted(by_dish)],
        off_menu_examples=len(off_menu),
        off_menu_rejected=sum(p.predicted is None for p in off_menu),
        chance_accuracy=chance,
        reasons=dict(sorted(Counter(p.reason for p in predictions).items())),
        confusions=[(t, w, c) for (t, w), c in confusions.most_common() if t and w],
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def results_markdown(name: str, report: Report, *, oracle_category: bool) -> str:
    """The table pasted into the README, rendered from the report's numbers."""
    o = report.overall
    setting = (
        "true category given (match stage alone)"
        if oracle_category
        else "category from the labels file where given"
    )
    lines = [
        f"### Menu matching: {name}",
        "",
        f"{o.examples} on-menu crops, {setting}. Top-1 accuracy {_pct(o.accuracy)}, "
        f"unknown rate {_pct(o.unknown_rate)}, accuracy when answered "
        f"{_pct(o.accuracy_when_answered)}; chance {_pct(report.chance_accuracy)}.",
    ]
    if report.off_menu_examples:
        lines.append(
            f"Off-menu crops correctly called unknown: {report.off_menu_rejected} of "
            f"{report.off_menu_examples}."
        )
    lines += [
        "",
        "| Dish | Crops | Correct | Unknown | Accuracy | Unknown rate | Accuracy when answered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report.dishes:
        flag = "" if r.reliable else " *"
        lines.append(
            f"| {r.name}{flag} | {r.examples} | {r.correct} | {r.unknown} | {_pct(r.accuracy)} "
            f"| {_pct(r.unknown_rate)} | {_pct(r.accuracy_when_answered)} |"
        )
    if report.confusions:
        lines += ["", "Most frequent confusions (true -> predicted):", ""]
        lines += [f"- {t} -> {w}: {c}" for t, w, c in report.confusions[:10]]
    lines += ["", f"\\* fewer than {MIN_EXAMPLES} crops: too few to trust."]
    return "\n".join(lines) + "\n"


def write_results(
    name: str,
    report: Report,
    config: dict[str, Any],
    out_dir: Path = RESULTS_DIR,
    *,
    oracle_category: bool,
) -> Path:
    """Write ``<name>.json`` and ``<name>.md`` to ``out_dir``; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "config": config,
        "oracle_category": oracle_category,
        "overall": asdict(report.overall),
        "chance_accuracy": report.chance_accuracy,
        "off_menu": {
            "examples": report.off_menu_examples,
            "rejected": report.off_menu_rejected,
        },
        "reasons": report.reasons,
        "confusions": [{"true": t, "predicted": w, "count": c} for t, w, c in report.confusions],
        "dishes": [asdict(r) | {"reliable": r.reliable} for r in report.dishes],
    }
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = results_markdown(name, report, oracle_category=oracle_category)
    (out_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    return json_path


def evaluate(
    labels: Path,
    name: str,
    out_dir: Path = RESULTS_DIR,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    min_margin: float = DEFAULT_MIN_MARGIN,
    data_dir: Path = Path("data"),
) -> Path:
    """Run SigLIP matching over ``labels`` and write the results; return the JSON path."""
    examples = load_examples(labels)
    embedder = SiglipEmbedder(model_id)

    def matcher_for(menu: DailyMenu, meal: str | None) -> Matcher:
        return MenuMatcher(
            menu, embedder, meal=meal, min_probability=min_probability, min_margin=min_margin
        )

    predictions = predict(examples, lambda ref: load_menu(ref, data_dir=data_dir), matcher_for)
    config = {
        "labels": str(labels),
        "model_id": model_id,
        "min_probability": min_probability,
        "min_margin": min_margin,
    }
    oracle = all(ex.category is None for ex in examples if ex.dish is not None)
    return write_results(name, score(predictions), config, out_dir, oracle_category=oracle)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True, help="labels CSV or JSON")
    parser.add_argument("--name", required=True, help="results file name, e.g. match-v1")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--min-probability", type=float, default=DEFAULT_MIN_PROBABILITY)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = evaluate(
        args.labels,
        args.name,
        args.out,
        model_id=args.model,
        min_probability=args.min_probability,
        min_margin=args.min_margin,
        data_dir=args.data_dir,
    )
    print(path.with_suffix(".md").read_text(encoding="utf-8"))
    print("written:", path, path.with_suffix(".md"))


if __name__ == "__main__":
    main()
