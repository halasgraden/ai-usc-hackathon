"""Auto-label plate photos with Grounding DINO + SAM 2.1 (offline, GPU; never in the live path).

Grounding DINO turns a text prompt into boxes, SAM 2.1 turns each box into a mask. Photo file
names carry the dish (``fries_p2_scraps_03.jpg``), so each image is prompted only for ``plate``,
``bowl`` and the one food it contains. That is far more reliable than prompting every class.

Everything except ``Autolabeler`` is plain NumPy and is unit-tested. ``Autolabeler`` needs the
model weights and a GPU and is only exercised in the Colab notebook.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from core.types import SEG_CLASSES, FoodCategory, ImageBGR, Mask
from labeling.masks import mask_to_polygon, yolo_seg_line

PILOT_ITEMS_PATH = Path(__file__).with_name("pilot_items.yaml")
EMPTY_ITEM = "empty"
PLATE, BOWL = "plate", "bowl"
STRUCTURE_PHRASES = {PLATE: "plate", BOWL: "bowl"}

Box = tuple[float, float, float, float]
"""x1, y1, x2, y2 in pixels."""


@dataclass(frozen=True, slots=True)
class PilotItem:
    category: FoodCategory
    phrase: str


@dataclass(frozen=True, slots=True)
class LabeledMask:
    class_name: str
    score: float
    mask: Mask


def load_pilot_items(path: Path = PILOT_ITEMS_PATH) -> dict[str, PilotItem]:
    raw = yaml.safe_load(path.read_text())
    return {
        slug: PilotItem(FoodCategory(v["category"]), str(v["phrase"])) for slug, v in raw.items()
    }


def parse_photo_name(path: Path | str) -> tuple[str, str]:
    """``fries_p2_scraps_03.jpg`` -> ``("fries", "fries_p2")`` (item slug, capture group)."""
    parts = Path(path).stem.lower().split("_")
    item = parts[0]
    group = "_".join(parts[:2]) if len(parts) > 1 else item
    return item, group


def split_items(item: str) -> list[str]:
    """``"asparagus+chicken"`` -> ``["asparagus", "chicken"]``; ``"empty"`` -> ``["empty"]``."""
    return [part for part in item.split("+") if part]


def build_prompt(item: str, items: dict[str, PilotItem]) -> tuple[str, dict[str, str]]:
    """Grounding DINO prompt for one photo, and a phrase -> class-name lookup.

    ``item`` may name several foods joined by ``+`` for a mixed plate. Raises KeyError for a
    slug that is neither ``empty`` nor in ``items``.
    """
    phrases = dict(STRUCTURE_PHRASES)
    for slug in split_items(item):
        if slug == EMPTY_ITEM:
            continue
        pilot = items[slug]
        phrases[pilot.phrase] = pilot.category.value
    prompt = " . ".join(phrases) + " ."
    return prompt, phrases


def resolve_label(text: str, lookup: dict[str, str]) -> str | None:
    """Map the phrase Grounding DINO returned to a class name (None if unrecognised).

    DINO may return a sub-phrase or a merge of several; the longest known phrase contained in
    the text wins, and a bare ``plate`` wins over nothing.
    """
    text = text.strip().lower()
    hits = [p for p in lookup if p in text]
    if not hits:
        return None
    return lookup[max(hits, key=len)]


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap_fraction(inner: Mask, outer: Mask) -> float:
    """Share of ``inner``'s pixels that lie inside ``outer``."""
    area = int(inner.sum())
    return float((inner & outer).sum()) / area if area else 0.0


def filled(mask: Mask) -> Mask:
    """Convex hull of ``mask``: the whole plate disc even where SAM left holes under the food."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return mask
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def filter_labels(
    labels: list[LabeledMask],
    *,
    min_on_plate: float = 0.5,
    max_plate_fraction: float = 0.92,
    min_area_px: int = 64,
    max_bowl_on_plate: float = 0.5,
    max_duplicate_overlap: float = 0.8,
) -> list[LabeledMask]:
    """Clean raw detections into training labels.

    * keep only the best ``plate`` (score x area, so a high-scoring sliver or a plate in the
      corner of the frame cannot beat the real one);
    * drop food masks that are mostly off the plate/bowl (cutlery, napkins, the table). The
      test uses the plate's convex hull, because SAM often leaves holes where food sits;
    * drop a ``bowl`` that sits mostly on the plate: that is a food pile DINO called a bowl;
    * drop a food mask that is the whole plate (Grounding DINO often boxes the plate for
      "fries");
    * drop a food mask that mostly overlaps a higher-scoring mask of the same class (the same
      spears boxed twice), so no pixel carries two instances;
    * drop slivers smaller than ``min_area_px``.
    """
    labels = [m for m in labels if int(m.mask.sum()) >= min_area_px]
    plates = sorted(
        (m for m in labels if m.class_name == PLATE),
        key=lambda m: -m.score * int(m.mask.sum()),
    )
    plate_region = filled(plates[0].mask) if plates else None
    bowls = [
        m
        for m in labels
        if m.class_name == BOWL
        and (plate_region is None or overlap_fraction(m.mask, plate_region) < max_bowl_on_plate)
    ]
    vessels = plates[:1] + bowls
    regions = [filled(v.mask) for v in vessels]
    out = list(vessels)
    foods: list[LabeledMask] = []
    for m in sorted(labels, key=lambda m: -m.score):
        if m.class_name in STRUCTURE_PHRASES:
            continue
        if not vessels:
            continue  # no plate found: a food mask has nothing to be validated against
        best_on = max(overlap_fraction(m.mask, r) for r in regions)
        if best_on < min_on_plate:
            continue
        if any(int(m.mask.sum()) > max_plate_fraction * int(r.sum()) for r in regions):
            continue
        if any(
            k.class_name == m.class_name
            and max(overlap_fraction(m.mask, k.mask), overlap_fraction(k.mask, m.mask))
            > max_duplicate_overlap
            for k in foods
        ):
            continue
        foods.append(m)
    return out + foods


def missing_items(item: str, labels: list[LabeledMask], items: dict[str, PilotItem]) -> list[str]:
    """Slugs named in ``item`` whose food class got no mask (an incompletely labelled photo).

    Training on such a photo teaches the model that the missed food is background, so the
    notebook leaves these photos out. Two slugs sharing a class (asparagus and carrots) are
    satisfied by one mask of that class.
    """
    found = {m.class_name for m in labels}
    return [
        slug
        for slug in split_items(item)
        if slug != EMPTY_ITEM and items[slug].category.value not in found
    ]


def to_yolo_seg_lines(labels: list[LabeledMask]) -> list[str]:
    """One Ultralytics segmentation label line per mask (skips masks with no usable outline)."""
    lines = []
    for m in labels:
        polygon = mask_to_polygon(m.mask)
        if polygon is not None:
            lines.append(yolo_seg_line(SEG_CLASSES.index(m.class_name), polygon))
    return lines


def split_groups(groups: list[str], val_fraction: float = 0.2, seed: int = 0) -> set[str]:
    """Pick the capture groups held out for validation, deterministically.

    Whole groups (e.g. one plate of fries) go to validation so no near-duplicate photo of a
    training plate is ever in the validation set. Every item with 2+ groups gets at least one.
    """
    unique = sorted(set(groups))

    def key(g: str) -> str:
        return hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()

    val: set[str] = set()
    by_item: dict[str, list[str]] = {}
    for g in unique:
        by_item.setdefault(g.split("_")[0], []).append(g)
    for members in by_item.values():
        if len(members) < 2:
            continue
        ranked = sorted(members, key=key)
        n_val = max(1, round(val_fraction * len(members)))
        val.update(ranked[:n_val])
    return val


class Autolabeler:
    """Grounding DINO + SAM 2.1 via Hugging Face ``transformers`` (needs a GPU).

    Not covered by unit tests: it needs the model weights. Check its output visually in the
    notebook before training on it.
    """

    def __init__(
        self,
        dino_id: str = "IDEA-Research/grounding-dino-base",
        sam_id: str = "facebook/sam2.1-hiera-small",
        box_threshold: float = 0.30,
        food_threshold: float = 0.25,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.food_threshold = food_threshold
        self.text_threshold = text_threshold
        self._dino_proc = AutoProcessor.from_pretrained(dino_id)
        self._dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device)
        self._sam_proc = Sam2Processor.from_pretrained(sam_id)
        self._sam = Sam2Model.from_pretrained(sam_id).to(self.device)

    def _detect(
        self, pil_image: Any, prompt: str, threshold: float | None = None
    ) -> tuple[list[Box], list[float], list[str]]:
        import torch

        thr = self.box_threshold if threshold is None else threshold
        inputs = self._dino_proc(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._dino(**inputs)
        kwargs: dict[str, Any] = {
            "text_threshold": self.text_threshold,
            "target_sizes": [pil_image.size[::-1]],
        }
        try:
            res = self._dino_proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=thr, **kwargs
            )[0]
        except TypeError:  # older transformers call it box_threshold
            res = self._dino_proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=thr, **kwargs
            )[0]
        texts = res.get("text_labels", res.get("labels", []))
        boxes: list[Box] = [(b[0], b[1], b[2], b[3]) for b in res["boxes"].cpu().tolist()]
        return boxes, [float(s) for s in res["scores"].cpu().tolist()], [str(t) for t in texts]

    def _segment(self, pil_image: Any, boxes: list[Box]) -> NDArray[np.bool_]:
        import torch

        inputs = self._sam_proc(
            images=pil_image, input_boxes=[[list(b) for b in boxes]], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._sam(**inputs, multimask_output=False)
        masks = self._sam_proc.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        return np.asarray(masks.reshape(len(boxes), *masks.shape[-2:]) > 0, dtype=np.bool_)

    def label(
        self, image_bgr: ImageBGR, item: str, items: dict[str, PilotItem]
    ) -> list[LabeledMask]:
        """Plate, bowl and food masks for one photo whose dish is ``item``."""
        from PIL import Image

        _, lookup = build_prompt(item, items)
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        # One single-phrase DINO pass per class: a prompt naming several foods weakens each
        # of them, and a single phrase means the class is known without reading DINO's text
        # labels (which can be clipped, blank, or a different length from the boxes).
        found: list[tuple[Box, float, str]] = []
        for phrase, cls in lookup.items():
            thr = self.box_threshold if phrase in STRUCTURE_PHRASES else self.food_threshold
            boxes, scores, _ = self._detect(pil, f"{phrase} .", thr)
            found.extend((b, sc, cls) for b, sc in zip(boxes, scores, strict=True))
        if not found:
            return []
        masks = self._segment(pil, [b for b, _, _ in found])
        raw = [LabeledMask(c, sc, m) for (_, sc, c), m in zip(found, masks, strict=True)]
        return filter_labels(raw)
