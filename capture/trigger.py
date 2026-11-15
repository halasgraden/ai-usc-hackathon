"""Decide when a plate is in the slot and which single frame to keep."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from core.types import CapturedFrame, ImageBGR


def sharpness(gray: NDArray[np.uint8] | NDArray[np.float32]) -> float:
    """Variance of the Laplacian of a grayscale image; higher means sharper.

    Uses the 4-neighbour Laplacian over the interior pixels. Among the frames of
    one still period the capture loop keeps the one with the highest value.

    Raises:
        ValueError: the image is not 2-D or is smaller than 3 x 3.
    """
    if gray.ndim != 2 or min(gray.shape) < 3:
        raise ValueError("sharpness needs a 2-D image of at least 3 x 3 pixels")
    g = gray.astype(np.float64)
    laplacian = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(laplacian.var())


class PlateTrigger:
    """Emit one frame per plate.

    Demo behaviour (plate set down under a phone on a stand): motion is seen,
    then the frame holds still for ``still_ms``; the sharpest frame of the still
    period is emitted once, and nothing more until motion is seen again.

    Args:
        motion_threshold: mean absolute frame difference (0-255) counted as motion.
        still_ms: how long the frame must hold still before a capture.
    """

    def __init__(self, motion_threshold: float = 4.0, still_ms: int = 300) -> None:
        self.motion_threshold = motion_threshold
        self.still_ms = still_ms

    def update(self, frame: ImageBGR, now: datetime) -> CapturedFrame | None:
        """Feed the next camera frame; returns a capture when one is ready."""
        raise NotImplementedError("hour 2-3")
