"""Decide when a plate is in the slot and which single frame to keep.

The phone looks straight down at the dish-return slot. The trigger is a small
state machine over cheap OpenCV operations on a downscaled, blurred grey copy
of each frame:

1. ``learning``: the first still frames become the picture of the empty slot
   (the rig must start with the slot empty).
2. ``empty``: waiting. The empty picture slowly follows lighting drift while
   the scene is still. When enough pixels differ from it, something arrived.
3. ``occupied``: wait for the scene to hold still (low frame-to-frame
   difference for ``still_frames`` frames and ``still_ms``), with the changed
   region clear of the image border (no arm reaching in, plate not half out of
   frame) and sharp enough (variance of the Laplacian). Then emit the sharpest
   still frame, once.
4. ``captured``: nothing more is emitted until the slot looks empty again for
   ``still_frames`` frames; then back to ``empty`` (re-armed).

A still scene that matches the empty picture never triggers, and a blank
(featureless) frame has zero sharpness, so it can never be emitted either.

All thresholds are UNTUNED defaults chosen by reasoning, not measured on the
rig; tune them on recorded dish-return footage before relying on them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from capture.scale import touches_border
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


def _gray_resized(frame: ImageBGR, width: int) -> NDArray[np.uint8]:
    """Grey copy no wider than ``width`` (aspect kept; INTER_AREA averages, so it denoises)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w > width:
        gray = cv2.resize(gray, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return np.asarray(gray, dtype=np.uint8)


class TriggerState(StrEnum):
    """Where the trigger is in the plate cycle (see the module docstring)."""

    LEARNING = "learning"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    CAPTURED = "captured"


class PlateTrigger:
    """Emit one frame per plate.

    Demo behaviour (plate set down under a phone on a stand): the slot changes
    from its empty picture, then the frame holds still for ``still_ms`` and
    ``still_frames`` frames; the sharpest frame of the still period is emitted
    once, and nothing more until the slot is empty again.

    Every threshold below is an untuned default.

    Args:
        motion_threshold: mean absolute frame-to-frame difference (0-255, on the
            downscaled grey frame) counted as motion.
        still_ms: how long the frame must hold still before a capture.
        still_frames: how many consecutive still frames are needed before a
            capture (and to learn the empty slot, and to re-arm).
        min_sharpness: variance of the Laplacian (on the grey frame scaled to at
            most ``sharpness_width`` px wide) a frame needs to be emitted.
        pixel_threshold: per-pixel absolute difference (0-255) from the empty
            picture that counts a pixel as changed.
        occupied_fraction: share of changed pixels at which the slot counts as
            occupied.
        empty_fraction: share of changed pixels below which the slot counts as
            empty again. Keep it below ``occupied_fraction`` (hysteresis).
        background_alpha: how fast the empty picture follows lighting drift
            (weight of each still empty frame, 0-1).
        require_clear_border: only emit when the changed region does not touch
            the image border (an arm in the frame, or a plate half out of view).
        work_width: width of the grey frame used for change and motion tests.
        sharpness_width: width of the grey frame used for ``sharpness``.
    """

    def __init__(
        self,
        motion_threshold: float = 4.0,
        still_ms: int = 300,
        *,
        still_frames: int = 5,
        min_sharpness: float = 50.0,
        pixel_threshold: int = 25,
        occupied_fraction: float = 0.05,
        empty_fraction: float = 0.01,
        background_alpha: float = 0.05,
        require_clear_border: bool = True,
        work_width: int = 160,
        sharpness_width: int = 640,
    ) -> None:
        if motion_threshold <= 0 or still_ms < 0 or still_frames < 1:
            raise ValueError("need motion_threshold > 0, still_ms >= 0, still_frames >= 1")
        if not 0.0 <= empty_fraction < occupied_fraction <= 1.0:
            raise ValueError("need 0 <= empty_fraction < occupied_fraction <= 1")
        if not 0.0 < background_alpha <= 1.0 or not 0 < pixel_threshold < 256:
            raise ValueError("need 0 < background_alpha <= 1 and 0 < pixel_threshold < 256")
        if work_width < 8 or sharpness_width < 8 or min_sharpness < 0:
            raise ValueError("need work_width, sharpness_width >= 8 and min_sharpness >= 0")
        self.motion_threshold = motion_threshold
        self.still_ms = still_ms
        self.still_frames = still_frames
        self.min_sharpness = min_sharpness
        self.pixel_threshold = pixel_threshold
        self.occupied_fraction = occupied_fraction
        self.empty_fraction = empty_fraction
        self.background_alpha = background_alpha
        self.require_clear_border = require_clear_border
        self.work_width = work_width
        self.sharpness_width = sharpness_width
        self.reset()

    def reset(self) -> None:
        """Forget everything, including the empty picture (relearned from the next frames)."""
        self.state = TriggerState.LEARNING
        self._background: NDArray[np.float32] | None = None
        self._previous: NDArray[np.float32] | None = None
        self._clear_still_run()
        self._clear_empty_run()

    def _clear_empty_run(self) -> None:
        self._empty_count = 0
        self._empty_since: datetime | None = None

    def _clear_still_run(self) -> None:
        self._still_count = 0
        self._still_since: datetime | None = None
        self._best: CapturedFrame | None = None

    def _small(self, frame: ImageBGR) -> NDArray[np.float32]:
        small = cv2.GaussianBlur(_gray_resized(frame, self.work_width), (5, 5), 0)
        return np.asarray(small, dtype=np.float32)

    def _changed_mask(self, small: NDArray[np.float32]) -> NDArray[np.bool_]:
        assert self._background is not None
        changed = (np.abs(small - self._background) > self.pixel_threshold).astype(np.uint8)
        # Opening removes isolated noisy pixels so they neither occupy the slot
        # nor make the changed region "touch the border".
        opened = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return np.asarray(opened, dtype=bool)

    def _still_long_enough(self, now: datetime) -> bool:
        if self._still_since is None or self._still_count < self.still_frames:
            return False
        return (now - self._still_since).total_seconds() * 1000.0 >= self.still_ms

    def update(self, frame: ImageBGR, now: datetime) -> CapturedFrame | None:
        """Feed the next camera frame; returns a capture when one is ready.

        The returned ``CapturedFrame`` holds a copy of the full-resolution
        frame; ``scale`` is None (the plate scale needs the segmentation mask,
        which the API computes downstream).

        Raises:
            ValueError: ``frame`` is not an H x W x 3 uint8 image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be an H x W x 3 uint8 BGR image")
        small = self._small(frame)
        previous = self._previous
        if previous is not None and previous.shape != small.shape:
            self.reset()  # resolution changed: the empty picture no longer applies
            previous = None
        self._previous = small
        moving = (
            previous is None or float(np.mean(np.abs(small - previous))) > self.motion_threshold
        )
        if moving:
            self._clear_still_run()
        else:
            self._still_count += 1
            if self._still_since is None:
                self._still_since = now

        if self.state is TriggerState.LEARNING:
            if self._still_long_enough(now):
                self._background = small.copy()
                self.state = TriggerState.EMPTY
                self._clear_still_run()
            return None

        changed = self._changed_mask(small)
        changed_fraction = float(changed.mean())

        if self.state is TriggerState.EMPTY:
            if changed_fraction >= self.occupied_fraction:
                self.state = TriggerState.OCCUPIED
                self._clear_still_run()
                if not moving:  # arrived between two frames and is already still
                    self._still_count, self._still_since = 1, now
            elif not moving:
                assert self._background is not None
                cv2.accumulateWeighted(small, self._background, self.background_alpha)
            if self.state is TriggerState.EMPTY:
                return None

        if self.state is TriggerState.CAPTURED:
            if moving or changed_fraction >= self.empty_fraction:
                self._clear_empty_run()
                return None
            self._empty_count += 1
            if self._empty_since is None:
                self._empty_since = now
            elapsed_ms = (now - self._empty_since).total_seconds() * 1000.0
            if self._empty_count >= self.still_frames and elapsed_ms >= self.still_ms:
                self.state = TriggerState.EMPTY  # re-armed for the next plate
                self._clear_still_run()
                self._clear_empty_run()
            return None

        # OCCUPIED
        if changed_fraction < self.empty_fraction:
            self.state = TriggerState.EMPTY  # removed before it was captured
            self._clear_still_run()
            return None
        if moving:
            return None
        if self.require_clear_border and touches_border(changed, margin_px=1):
            self._clear_still_run()  # something is still crossing the edge of the view
            return None
        score = sharpness(_gray_resized(frame, self.sharpness_width))
        if self._best is None or score > self._best.sharpness:
            self._best = CapturedFrame(
                image=frame.copy(), captured_at=now, sharpness=score, scale=None
            )
        if not self._still_long_enough(now) or self._best.sharpness < self.min_sharpness:
            return None
        captured = self._best
        self.state = TriggerState.CAPTURED
        self._clear_still_run()
        self._clear_empty_run()
        return captured
