"""Privacy backstop: refuse any frame that shows a face.

The detector is OpenCV's YuNet (``cv2.FaceDetectorYN``) with the
``face_detection_yunet_2023mar.onnx`` weights from opencv_zoo. The weights are
not bundled with the OpenCV wheels; put the file at ``LEFTOVER_FACE_MODEL``
(default ``models/face_detection_yunet_2023mar.onnx``, see ``models/FACE_MODEL.md``).

The guard FAILS CLOSED: if the model file is missing, cannot be loaded, or the
detector errors on a frame, the frame is treated as containing a face and is
dropped, and ``FaceCheck.reason`` says why. A rig without a working detector
therefore uploads nothing, rather than uploading unchecked frames.

No detection accuracy has been measured for this setup (camera straight down
over a dish return); the score threshold is untuned.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from core.types import ImageBGR

FACE_MODEL_ENV = "LEFTOVER_FACE_MODEL"
"""Environment variable holding the path of the YuNet ONNX file."""

DEFAULT_FACE_MODEL = Path("models/face_detection_yunet_2023mar.onnx")
"""Used when ``LEFTOVER_FACE_MODEL`` is unset; relative to the working directory."""

FaceDetector = Callable[[ImageBGR], int]
"""Returns how many faces are in a BGR frame; raising counts as a failure (fail closed)."""


@dataclass(frozen=True, slots=True)
class FaceCheck:
    """Outcome of checking one frame. ``has_face`` True means: drop the frame, unstored."""

    has_face: bool
    reason: str


def yunet_detector(
    model_path: Path | str,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    max_side: int = 640,
) -> FaceDetector:
    """Load YuNet and return a detector that counts faces in a frame.

    Frames larger than ``max_side`` on their long side are downscaled first,
    for speed (very small faces may then be missed; untested). ``score_threshold``
    0.6 is set lowish on purpose, to err towards dropping frames; it is not tuned.

    Raises:
        FileNotFoundError: the model file does not exist.
        cv2.error: OpenCV cannot load the file as a YuNet model.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YuNet model not found at {path}")
    net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, nms_threshold)

    def detect(image: ImageBGR) -> int:
        h, w = image.shape[:2]
        factor = min(1.0, max_side / max(h, w))
        if factor < 1.0:
            size = (max(1, round(w * factor)), max(1, round(h * factor)))
            image = np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), np.uint8)
        net.setInputSize((image.shape[1], image.shape[0]))
        _, faces = net.detect(image)
        return 0 if faces is None else len(faces)

    return detect


class FaceGuard:
    """Checks frames for faces; with no detector, every frame counts as a face.

    Args:
        detector: counts faces in a frame. None means the detector is
            unavailable, and every frame is refused.
        unavailable_reason: why ``detector`` is None, repeated in each refusal.
    """

    def __init__(self, detector: FaceDetector | None, unavailable_reason: str = "") -> None:
        self.detector = detector
        self.unavailable_reason = unavailable_reason or "no face detector configured"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FaceGuard:
        """YuNet from ``LEFTOVER_FACE_MODEL``; never raises (a load failure fails closed)."""
        env = os.environ if env is None else env
        path = Path(env.get(FACE_MODEL_ENV, "").strip() or DEFAULT_FACE_MODEL)
        try:
            return cls(yunet_detector(path))
        except FileNotFoundError:
            return cls(
                None,
                f"YuNet model missing at {path}; download face_detection_yunet_2023mar.onnx "
                f"from opencv_zoo and set {FACE_MODEL_ENV}",
            )
        except Exception as exc:  # cv2.error and anything else from the loader
            return cls(None, f"YuNet model at {path} failed to load: {exc}")

    @property
    def available(self) -> bool:
        return self.detector is not None

    def check(self, image: ImageBGR) -> FaceCheck:
        """Check one frame. Any doubt (no detector, bad frame, detector error) is a face."""
        if self.detector is None:
            return FaceCheck(True, f"fail closed: {self.unavailable_reason}")
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            return FaceCheck(True, "fail closed: frame is not an H x W x 3 image")
        if image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            return FaceCheck(True, "fail closed: frame is empty or not uint8")
        try:
            count = int(self.detector(image))
        except Exception as exc:
            return FaceCheck(True, f"fail closed: face detector error: {exc}")
        if count > 0:
            return FaceCheck(True, f"{count} face(s) detected")
        return FaceCheck(False, "no face detected")


@lru_cache(maxsize=1)
def default_guard() -> FaceGuard:
    """The process-wide guard built from the environment (loaded once)."""
    return FaceGuard.from_env()


def contains_face(image: ImageBGR, guard: FaceGuard | None = None) -> bool:
    """True if a face is detected. Frames that return True are dropped unstored.

    The camera points straight down at the tray, so this should never fire; it
    exists so that a knocked or mis-aimed phone cannot leak a face into storage.
    Also True when the detector is unavailable or fails (fail closed); use
    ``FaceGuard.check`` to get the reason.
    """
    return (guard or default_guard()).check(image).has_face
