"""Open the phone camera as an OpenCV video source.

Main path: the phone appears as a webcam (Continuity Camera, DroidCam, Iriun)
and is opened by device index, or streams over the network (IP Webcam) and is
opened by URL. Both go through ``cv2.VideoCapture``, so the capture loop does
not care which one is in use. A path to a recorded video also works, which is
handy for replaying a dish-return clip while tuning the trigger.

Frames are only ever held in memory; nothing here writes to disk.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import cv2
import numpy as np

from core.types import ImageBGR


class FrameSource(Protocol):
    """Anything that yields frames; ``cv2.VideoCapture`` wrappers and test fakes."""

    def read(self) -> ImageBGR | None:
        """Return the next frame, or None when the source has no frame ready."""
        ...

    def close(self) -> None: ...


class VideoCaptureLike(Protocol):
    """The part of ``cv2.VideoCapture`` this module uses (fakes implement it in tests)."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


def parse_source(spec: str) -> int | str:
    """Interpret ``LEFTOVER_CAMERA_SOURCE``: digits are a device index, else a URL.

    >>> parse_source("0")
    0
    >>> parse_source("http://192.168.1.20:8080/video")
    'http://192.168.1.20:8080/video'
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty camera source")
    return int(spec) if spec.isdigit() else spec


class VideoCaptureSource:
    """A ``FrameSource`` over an opened ``cv2.VideoCapture`` (or anything shaped like one).

    ``read`` returns None for a failed grab or for anything that is not an
    H x W x 3 uint8 image, so the caller only ever sees usable BGR frames.
    """

    def __init__(self, capture: VideoCaptureLike) -> None:
        self._capture = capture
        self._closed = False

    def read(self) -> ImageBGR | None:
        if self._closed:
            return None
        ok, frame = self._capture.read()
        if not ok or not isinstance(frame, np.ndarray):
            return None
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            return None
        return frame

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()


def open_source(
    spec: str, opener: Callable[[int | str], VideoCaptureLike] | None = None
) -> FrameSource:
    """Open a camera by device index or stream URL.

    Args:
        spec: ``LEFTOVER_CAMERA_SOURCE``, see ``parse_source``.
        opener: builds the capture object; defaults to ``cv2.VideoCapture``.
            Tests inject a fake here.

    Raises:
        ValueError: ``spec`` is empty.
        RuntimeError: the device or stream cannot be opened.
    """
    target = parse_source(spec)
    capture = (opener or cv2.VideoCapture)(target)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"cannot open camera source {target!r}: check LEFTOVER_CAMERA_SOURCE "
            "(device index such as 0, or a stream URL such as http://<phone-ip>:8080/video)"
        )
    return VideoCaptureSource(capture)
