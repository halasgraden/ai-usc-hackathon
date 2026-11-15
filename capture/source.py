"""Open the phone camera as an OpenCV video source.

Main path: the phone appears as a webcam (Continuity Camera, DroidCam, Iriun)
and is opened by device index, or streams over the network (IP Webcam) and is
opened by URL. Both go through ``cv2.VideoCapture``, so the capture loop does
not care which one is in use.
"""

from __future__ import annotations

from typing import Protocol

from core.types import ImageBGR


class FrameSource(Protocol):
    """Anything that yields frames; ``cv2.VideoCapture`` wrappers and test fakes."""

    def read(self) -> ImageBGR | None:
        """Return the next frame, or None when the source has no frame ready."""
        ...

    def close(self) -> None: ...


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


def open_source(spec: str) -> FrameSource:
    """Open a camera by device index or stream URL.

    Raises:
        RuntimeError: the device or stream cannot be opened.
    """
    raise NotImplementedError("hour 1-2: wrap cv2.VideoCapture(parse_source(spec))")
