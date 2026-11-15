"""Privacy backstop: refuse any frame that shows a face."""

from __future__ import annotations

from core.types import ImageBGR


def contains_face(image: ImageBGR) -> bool:
    """True if a face is detected. Frames that return True are dropped unstored.

    The camera points straight down at the tray, so this should never fire; it
    exists so that a knocked or mis-aimed phone cannot leak a face into storage.
    """
    raise NotImplementedError("hour 6+: OpenCV YuNet face detector, fail closed on error")
