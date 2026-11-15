"""Copy-paste and photometric augmentation for plate photos.

Training-only. Cut labeled food out of one photo (``extract_cutouts``), paste it onto another
plate (``composite``), and vary lighting (``photometric_jitter``). That makes mixed plates and
many food positions out of single-food photos.

Rules that keep evaluation honest (the caller enforces them, see ``build_composites``):

* cutouts come only from training photos, never from a validation group;
* composites are written to the training split only;
* validation images are always real photos.

Hue is never shifted: colour is what separates carrots, asparagus and fries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from core.types import ImageBGR, Mask
from labeling.autolabel import BOWL, PLATE, LabeledMask, overlap_fraction


@dataclass(frozen=True, slots=True)
class Cutout:
    """One food instance cropped to its mask bounding box."""

    class_name: str
    image: ImageBGR
    mask: Mask
    plate_px: float
    """Square root of the source plate's area in pixels (0 if unknown); used to match scale."""


def _best_plate(labels: Sequence[LabeledMask]) -> LabeledMask | None:
    plates = [m for m in labels if m.class_name == PLATE]
    return max(plates, key=lambda m: m.score) if plates else None


def extract_cutouts(
    image: ImageBGR, labels: Sequence[LabeledMask], *, min_area_px: int = 400
) -> list[Cutout]:
    """Crop every food mask (not plate or bowl) out of ``image``."""
    plate = _best_plate(labels)
    plate_px = float(np.sqrt(plate.mask.sum())) if plate is not None else 0.0
    out: list[Cutout] = []
    for m in labels:
        if m.class_name in (PLATE, BOWL) or int(m.mask.sum()) < min_area_px:
            continue
        ys, xs = np.nonzero(m.mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out.append(
            Cutout(
                m.class_name,
                image[y0:y1, x0:x1].copy(),
                m.mask[y0:y1, x0:x1].copy(),
                plate_px,
            )
        )
    return out


def _transform(cutout: Cutout, angle: float, scale: float) -> tuple[ImageBGR, Mask]:
    """Rotate and scale a cutout on a canvas big enough not to clip it; re-crop to its mask."""
    h, w = cutout.mask.shape
    side = int(np.ceil(np.hypot(h, w) * scale)) + 4
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += side / 2 - w / 2
    matrix[1, 2] += side / 2 - h / 2
    img = cv2.warpAffine(cutout.image, matrix, (side, side), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(
        cutout.mask.astype(np.uint8), matrix, (side, side), flags=cv2.INTER_NEAREST
    ).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.asarray(img[:1, :1], dtype=np.uint8), np.asarray(mask[:1, :1], dtype=np.bool_)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (
        np.asarray(img[y0:y1, x0:x1], dtype=np.uint8),
        np.asarray(mask[y0:y1, x0:x1], dtype=np.bool_),
    )


def _placement(
    shape: tuple[int, int], mask: Mask, cx: int, cy: int
) -> tuple[slice, slice, slice, slice] | None:
    """Canvas and patch slices for pasting ``mask`` centred at (cx, cy); None if off-canvas."""
    height, width = shape
    h, w = mask.shape
    x0, y0 = cx - w // 2, cy - h // 2
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, width), min(y0 + h, height)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (
        slice(cy0, cy1),
        slice(cx0, cx1),
        slice(cy0 - y0, cy1 - y0),
        slice(cx0 - x0, cx1 - x0),
    )


def composite(
    base: ImageBGR,
    base_labels: Sequence[LabeledMask],
    bank: Sequence[Cutout],
    rng: np.random.Generator,
    *,
    n_foods: tuple[int, int] = (1, 3),
    jitter: float = 0.2,
    max_overlap: float = 0.3,
    min_on_plate: float = 0.9,
    tries: int = 25,
    feather: float = 1.5,
) -> tuple[ImageBGR, list[LabeledMask]]:
    """Paste random cutouts onto the plate of ``base`` and return the new image and labels.

    Later pastes occlude earlier ones; labels are trimmed to what stays visible. Raises
    ValueError if ``base_labels`` has no plate.
    """
    plate = _best_plate(base_labels)
    if plate is None:
        raise ValueError("base image needs a plate label to paste onto")
    base_plate_px = float(np.sqrt(plate.mask.sum()))
    canvas = base.copy()
    labels = [LabeledMask(m.class_name, m.score, m.mask.copy()) for m in base_labels]
    plate_ys, plate_xs = np.nonzero(plate.mask)
    if len(bank) == 0:
        return canvas, labels

    wanted = int(rng.integers(n_foods[0], n_foods[1] + 1))
    pasted: list[LabeledMask] = []
    for _ in range(wanted):
        cut = bank[int(rng.integers(len(bank)))]
        for _attempt in range(tries):
            ratio = base_plate_px / cut.plate_px if cut.plate_px > 0 else 1.0
            scale = ratio * float(rng.uniform(1 - jitter, 1 + jitter))
            img, mask = _transform(cut, float(rng.uniform(0, 360)), scale)
            k = int(rng.integers(len(plate_xs)))
            spot = _placement(canvas.shape[:2], mask, int(plate_xs[k]), int(plate_ys[k]))
            if spot is None:
                continue
            cy, cx, py, px = spot
            full: Mask = np.zeros(canvas.shape[:2], dtype=bool)
            full[cy, cx] = mask[py, px]
            area = int(full.sum())
            if area < 64 or area < int(mask.sum()) * 0.9:
                continue  # clipped by the image edge
            if overlap_fraction(full, plate.mask) < min_on_plate:
                continue
            if any(overlap_fraction(full, p.mask) > max_overlap for p in pasted):
                continue
            alpha = np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather), 0, 1)
            alpha3 = alpha[py, px, None]
            region = canvas[cy, cx].astype(np.float32)
            blended = alpha3 * img[py, px].astype(np.float32) + (1 - alpha3) * region
            canvas[cy, cx] = np.clip(blended, 0, 255).astype(np.uint8)
            for existing in (*labels, *pasted):
                if existing.class_name not in (PLATE, BOWL):
                    existing.mask[full] = False  # now hidden under the new paste
            pasted.append(LabeledMask(cut.class_name, 1.0, full))
            break
    kept = [m for m in labels if m.class_name in (PLATE, BOWL) or int(m.mask.sum()) >= 64]
    kept += [m for m in pasted if int(m.mask.sum()) >= 64]
    return canvas, kept


def photometric_jitter(
    image: ImageBGR,
    rng: np.random.Generator,
    *,
    brightness: float = 0.2,
    contrast: float = 0.2,
    temperature: float = 0.08,
    gamma: float = 0.2,
    noise_sigma: float = 3.0,
    blur_prob: float = 0.2,
) -> ImageBGR:
    """Vary lighting like a different bulb, time of day or camera. Never shifts hue."""
    img = image.astype(np.float32)
    t = float(rng.uniform(-temperature, temperature))
    img[..., 0] *= 1 - t  # blue
    img[..., 2] *= 1 + t  # red
    mean = float(img.mean())
    img = (img - mean) * (1 + float(rng.uniform(-contrast, contrast))) + mean
    img += float(rng.uniform(-brightness, brightness)) * 255.0
    g = float(np.exp(rng.uniform(-gamma, gamma)))
    img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, g)
    if noise_sigma > 0:
        img += rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    out: ImageBGR = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < blur_prob:
        out = np.asarray(
            cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.5))), dtype=np.uint8
        )
    return np.asarray(out, dtype=np.uint8)


def build_composites(
    bases: Sequence[tuple[ImageBGR, Sequence[LabeledMask]]],
    bank: Sequence[Cutout],
    n: int,
    seed: int = 0,
    **kwargs: float | int | tuple[int, int],
) -> list[tuple[ImageBGR, list[LabeledMask]]]:
    """Make ``n`` composites with lighting jitter from empty-plate ``bases``.

    Pass only training photos in ``bases`` and ``bank``.
    """
    rng = np.random.default_rng(seed)
    out: list[tuple[ImageBGR, list[LabeledMask]]] = []
    for _ in range(n):
        image, labels = bases[int(rng.integers(len(bases)))]
        new_image, new_labels = composite(image, labels, bank, rng, **kwargs)  # type: ignore[arg-type]
        out.append((photometric_jitter(new_image, rng), new_labels))
    return out
