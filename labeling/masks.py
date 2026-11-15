"""Mask geometry shared by the labelers: box masks to full-image masks to YOLO polygons.

Uses OpenCV (install the ``vision`` extra, or ``opencv-python-headless``).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from core.types import Mask

BBox = tuple[float, float, float, float]
"""``(x, y, w, h)`` as fractions of image width and height (FiftyOne's convention)."""
Polygon = list[tuple[float, float]]
"""Vertices as ``(x, y)`` fractions of image width and height."""


def place_box_mask(box_mask: NDArray[np.bool_], bbox: BBox, width: int, height: int) -> Mask:
    """Paste a FiftyOne-style detection mask into a full-size mask.

    FiftyOne stores an instance mask cropped to its bounding box. The crop is
    resized (nearest neighbour) to the box's pixel size and placed at the box.

    Raises:
        ValueError: the mask is not 2-D or the image size is not positive.
    """
    if box_mask.ndim != 2:
        raise ValueError("box mask must be 2-D")
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    bx, by, bw, bh = bbox
    x0, y0 = max(0, round(bx * width)), max(0, round(by * height))
    x1, y1 = min(width, round((bx + bw) * width)), min(height, round((by + bh) * height))
    full: Mask = np.zeros((height, width), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return full
    resized = cv2.resize(
        box_mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
    )
    full[y0:y1, x0:x1] = resized.astype(bool)
    return full


def mask_to_polygon(
    mask: Mask, *, min_area_px: float = 64.0, epsilon_frac: float = 0.002
) -> Polygon | None:
    """Outline of the largest blob in ``mask``, normalized to 0-1.

    Returns None when the mask is empty, the blob is smaller than ``min_area_px``,
    or simplification leaves fewer than 3 vertices.
    """
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None
    simplified = cv2.approxPolyDP(largest, epsilon_frac * cv2.arcLength(largest, True), True)
    if len(simplified) < 3:
        return None
    return [(float(p[0][0]) / width, float(p[0][1]) / height) for p in simplified]


def yolo_seg_line(class_index: int, polygon: Polygon) -> str:
    """One Ultralytics segmentation label line: ``<class> x1 y1 x2 y2 ...``."""
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{class_index} {coords}"
