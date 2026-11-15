from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from core.types import DishMatch, FoodCategory, FoodRegion
from eval.matching import (
    MIN_EXAMPLES,
    Example,
    Prediction,
    load_examples,
    load_image,
    load_mask,
    load_menu,
    predict,
    results_markdown,
    score,
    true_category,
    write_results,
)
from menu.matcher import MatchDetail, MatchReason
from menu.models import DailyMenu
from tests.conftest import VILLAGE_SAMPLE

C = FoodCategory


def P(true: str | None, pred: str | None, n: int = 2, reason: str = "matched") -> Prediction:
    return Prediction(true_dish=true, predicted=pred, reason=reason, n_candidates=n, score=0.5)


def test_score_counts_accuracy_unknowns_and_answered() -> None:
    preds = [
        P("Rice Pilaf", "Rice Pilaf", n=1),
        P("Roast Chicken", "Roast Chicken"),
        P("Roast Chicken", "Glazed Chicken"),
        P("Roast Chicken", None, reason="low_margin"),
        P(None, None, n=0, reason="no_candidates"),
        P(None, "Glazed Chicken"),
    ]
    report = score(preds)
    o = report.overall
    assert (o.examples, o.correct, o.unknown) == (4, 2, 1)
    assert o.accuracy == pytest.approx(0.5)
    assert o.unknown_rate == pytest.approx(0.25)
    assert o.accuracy_when_answered == pytest.approx(2 / 3)
    chicken = next(d for d in report.dishes if d.name == "Roast Chicken")
    assert chicken.accuracy == pytest.approx(1 / 3)
    assert (report.off_menu_examples, report.off_menu_rejected) == (2, 1)
    assert report.chance_accuracy == pytest.approx((1 + 0.5 * 3) / 4)
    assert report.confusions == [("Roast Chicken", "Glazed Chicken", 1)]
    assert report.reasons == {"low_margin": 1, "matched": 4, "no_candidates": 1}


def test_score_all_unknown_has_no_answered_accuracy() -> None:
    report = score([P("Rice Pilaf", None, n=0)])
    assert report.overall.accuracy_when_answered is None
    assert report.chance_accuracy is None


def test_score_needs_on_menu_examples() -> None:
    with pytest.raises(ValueError):
        score([P(None, None)])


def test_results_markdown_renders_computed_numbers_and_flags() -> None:
    report = score([P("Rice Pilaf", "Rice Pilaf")] * MIN_EXAMPLES + [P("Roast Chicken", None)])
    md = results_markdown("m1", report, oracle_category=True)
    assert "true category given" in md
    assert f"| Rice Pilaf | {MIN_EXAMPLES} | {MIN_EXAMPLES} | 0 | 100.0% | 0.0% | 100.0% |" in md
    assert "| Roast Chicken * | 1 | 0 | 1 | 0.0% | 100.0% | n/a |" in md
    assert "Off-menu" not in md


def test_write_results_writes_json_and_md(tmp_path: Path) -> None:
    report = score([P("Rice Pilaf", "Rice Pilaf"), P(None, None)])
    path = write_results("m1", report, {"model_id": "x"}, tmp_path, oracle_category=False)
    data = json.loads(path.read_text())
    assert data["overall"]["accuracy"] == 1.0
    assert data["off_menu"] == {"examples": 1, "rejected": 1}
    assert data["dishes"][0]["reliable"] is False
    assert data["config"] == {"model_id": "x"}
    assert (tmp_path / "m1.md").read_text().startswith("### Menu matching: m1")


def test_load_examples_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "crop,dish,menu,date,meal,category,mask\n"
        "a.png,Rice Pilaf,menu.json,,Dinner,,\n"
        "b.png,unknown,,2026-11-14,,poultry,b_mask.png\n"
    )
    a, b = load_examples(csv_path)
    assert a == Example(tmp_path / "a.png", "Rice Pilaf", str(tmp_path / "menu.json"), "Dinner")
    assert b.dish is None and b.menu_ref == "2026-11-14" and b.category is C.POULTRY
    assert b.mask == tmp_path / "b_mask.png"

    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps([{"crop": "c.png", "dish": "", "date": "2026-11-14"}]))
    (c,) = load_examples(json_path)
    assert c.dish is None and c.meal is None


@pytest.mark.parametrize(
    "body",
    ["crop,dish\na.png,Rice Pilaf\n", "crop,dish,menu,category\na.png,x,m.json,pizza\n"],
)
def test_load_examples_rejects_bad_rows(tmp_path: Path, body: str) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_examples(path)


def test_load_menu_from_raw_file_dump_and_cached_date(tmp_path: Path) -> None:
    raw = load_menu(str(VILLAGE_SAMPLE))
    assert raw.day == date(2026, 11, 14) and raw.meal("Dinner") is not None
    dump = tmp_path / "menu.json"
    dump.write_text(raw.model_dump_json())
    assert load_menu(str(dump)) == raw
    cached = tmp_path / "menus" / "university-village" / "2026-11-14.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(VILLAGE_SAMPLE.read_text())
    assert load_menu("2026-11-14", data_dir=tmp_path).day == date(2026, 11, 14)


def test_true_category(categorized: DailyMenu) -> None:
    ex = Example(Path("a.png"), "Rice Pilaf", "m", meal="Dinner")
    assert true_category(ex, categorized) is C.RICE_GRAINS
    override = Example(Path("a.png"), "Rice Pilaf", "m", category=C.BREAD)
    assert true_category(override, categorized) is C.BREAD
    assert true_category(Example(Path("a.png"), None, "m"), categorized) is None
    with pytest.raises(ValueError):
        true_category(Example(Path("a.png"), "Lobster Thermidor", "m"), categorized)


def test_load_image_and_mask(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red
    pil.fromarray(rgb).save(tmp_path / "a.png")
    bgr = load_image(tmp_path / "a.png")
    assert bgr.shape == (4, 6, 3) and bgr[0, 0, 2] == 200 and bgr[0, 0, 0] == 0
    m = np.zeros((4, 6), dtype=np.uint8)
    m[1, 1] = 255
    pil.fromarray(m).save(tmp_path / "m.png")
    mask = load_mask(tmp_path / "m.png", (4, 6))
    assert mask.sum() == 1 and mask[1, 1]
    assert load_mask(None, (2, 3)).all()
    with pytest.raises(ValueError):
        load_mask(tmp_path / "m.png", (5, 6))


class FakeMatcher:
    def __init__(self) -> None:
        self.regions: list[FoodRegion] = []

    def explain(self, image: NDArray[np.uint8], region: FoodRegion) -> MatchDetail:
        self.regions.append(region)
        name = "Rice Pilaf" if region.category is C.RICE_GRAINS else None
        reason = MatchReason.SINGLE_CANDIDATE if name else MatchReason.NO_CANDIDATES
        return MatchDetail(
            DishMatch(name, 1.0 if name else 0.0), reason, (name,) if name else (), ()
        )


def test_predict_builds_menus_and_matchers_once(categorized: DailyMenu) -> None:
    examples = [
        Example(Path("a.png"), "Rice Pilaf", "m", meal="Dinner"),
        Example(Path("b.png"), "Rice Pilaf", "m", meal="Dinner"),
        Example(Path("c.png"), None, "m", meal="Dinner", category=C.ICE_CREAM),
    ]
    menu_calls: list[str] = []
    matcher_calls: list[str | None] = []
    fake = FakeMatcher()

    def menus(ref: str) -> DailyMenu:
        menu_calls.append(ref)
        return categorized

    def matcher_for(menu: DailyMenu, meal: str | None) -> FakeMatcher:
        matcher_calls.append(meal)
        return fake

    preds = predict(
        examples,
        menus,
        matcher_for,
        image_loader=lambda p: np.zeros((3, 3, 3), dtype=np.uint8),
    )
    assert menu_calls == ["m"] and matcher_calls == ["Dinner"]
    assert [p.predicted for p in preds] == ["Rice Pilaf", "Rice Pilaf", None]
    assert preds[2].reason == "no_candidates" and preds[0].n_candidates == 1
    assert fake.regions[0].category is C.RICE_GRAINS and fake.regions[0].mask.all()


def test_predict_off_menu_without_category_fails(categorized: DailyMenu) -> None:
    with pytest.raises(ValueError):
        predict(
            [Example(Path("a.png"), None, "m")],
            lambda ref: categorized,
            lambda menu, meal: FakeMatcher(),
            image_loader=lambda p: np.zeros((3, 3, 3), dtype=np.uint8),
        )
