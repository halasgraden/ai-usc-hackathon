"""Live segmentation with ONNX Runtime (no PyTorch on the rig).

Targets the file written by ``vision.export_onnx`` (Ultralytics YOLO-seg, opset 18):

* input  ``images``  (1, 3, S, S)  float32 RGB in [0, 1], letterboxed
* output ``output0`` (1, 4 + C + M, N)  box (cx, cy, w, h in letterbox pixels), C class
  scores (already sigmoid), M mask coefficients, for N anchors
* output ``output1`` (1, M, S/4, S/4)  mask prototypes

There is no NMS inside the graph, so it is done here. The pure functions
(``letterbox``, ``decode``) take plain arrays so they can be tested without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from core.types import SEG_CLASSES, FoodCategory, FoodRegion, ImageBGR, Mask

PAD_VALUE = 114
"""Grey used for letterbox padding (matches Ultralytics training)."""
MASK_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """How an image was fitted into the square model input."""

    scale: float
    pad_x: int
    pad_y: int
    orig_h: int
    orig_w: int


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected instance in original-image coordinates."""

    class_index: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask: Mask


class _Session(Protocol):
    def get_inputs(self) -> Sequence[Any]: ...

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[NDArray[np.float32]]: ...


def letterbox(image: ImageBGR, size: int) -> tuple[NDArray[np.float32], LetterboxInfo]:
    """Fit into ``size`` x ``size`` with grey padding; return (1, 3, S, S) RGB in [0, 1]."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    rgb = canvas[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    return tensor, LetterboxInfo(scale, pad_x, pad_y, h, w)


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _assemble_mask(
    coeffs: NDArray[np.float32],
    protos: NDArray[np.float32],
    box_lb: NDArray[np.float32],
    info: LetterboxInfo,
    size: int,
) -> Mask:
    """Combine prototypes into one instance mask at the original image size."""
    n_proto, ph, pw = protos.shape
    low = _sigmoid(coeffs @ protos.reshape(n_proto, -1)).reshape(ph, pw)
    # Work at letterbox resolution, zero outside the box, then undo the padding.
    full = cv2.resize(low, (size, size), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = (round(float(v)) for v in box_lb)
    keep = np.zeros_like(full, dtype=bool)
    keep[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = True
    full = np.where(keep, full, 0.0)
    new_w, new_h = round(info.orig_w * info.scale), round(info.orig_h * info.scale)
    crop = full[info.pad_y : info.pad_y + new_h, info.pad_x : info.pad_x + new_w]
    orig = cv2.resize(crop, (info.orig_w, info.orig_h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(orig > MASK_THRESHOLD, dtype=np.bool_)


def decode(
    output0: NDArray[np.float32],
    output1: NDArray[np.float32],
    info: LetterboxInfo,
    class_names: Sequence[str] = SEG_CLASSES,
    confidence: float = 0.25,
    iou: float = 0.7,
    max_det: int = 50,
) -> list[Detection]:
    """Turn raw network output into detections (class-aware NMS, masks at image size)."""
    n_cls = len(class_names)
    preds = output0[0].T  # (N, 4 + C + M)
    protos = output1[0]
    n_proto = protos.shape[0]
    if preds.shape[1] != 4 + n_cls + n_proto:
        raise ValueError(
            f"output has {preds.shape[1]} channels, expected 4 + {n_cls} classes + {n_proto} masks"
        )
    size = protos.shape[1] * 4

    scores = preds[:, 4 : 4 + n_cls]
    cls = scores.argmax(axis=1)
    conf = scores[np.arange(len(scores)), cls]
    keep = conf >= confidence
    if not keep.any():
        return []
    preds, cls, conf = preds[keep], cls[keep], conf[keep]

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # Class-aware NMS: shift each class far apart so boxes of different classes never overlap.
    shifted = boxes + (cls[:, None] * (size + 1)).astype(np.float32)
    xywh = np.stack(
        [
            shifted[:, 0],
            shifted[:, 1],
            shifted[:, 2] - shifted[:, 0],
            shifted[:, 3] - shifted[:, 1],
        ],
        axis=1,
    )
    picked = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), confidence, iou)
    order = [int(i) for i in np.asarray(picked).reshape(-1)][:max_det]

    out: list[Detection] = []
    for i in order:
        mask = _assemble_mask(preds[i, 4 + n_cls :], protos, boxes[i], info, size)
        bx = boxes[i]
        x1 = float(np.clip((bx[0] - info.pad_x) / info.scale, 0, info.orig_w))
        y1 = float(np.clip((bx[1] - info.pad_y) / info.scale, 0, info.orig_h))
        x2 = float(np.clip((bx[2] - info.pad_x) / info.scale, 0, info.orig_w))
        y2 = float(np.clip((bx[3] - info.pad_y) / info.scale, 0, info.orig_h))
        out.append(
            Detection(int(cls[i]), class_names[int(cls[i])], float(conf[i]), (x1, y1, x2, y2), mask)
        )
    return sorted(out, key=lambda d: d.confidence, reverse=True)


class Segmenter:
    """Segments the plate, bowls and food regions in one frame.

    Args:
        model_path: ONNX file produced by ``vision.export_onnx``.
        confidence: minimum detection confidence to keep a region.
        iou: NMS overlap threshold.
        session: inject an ONNX Runtime-like session (tests); loaded lazily from
            ``model_path`` when omitted.
    """

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.25,
        iou: float = 0.7,
        session: _Session | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self._session = session
        self._input_size: int | None = None

    def _get_session(self) -> _Session:
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        return self._session

    def _size(self, session: _Session) -> int:
        if self._input_size is None:
            shape = session.get_inputs()[0].shape
            self._input_size = int(shape[2])
        return self._input_size

    def detect(self, image: ImageBGR) -> list[Detection]:
        """All instances (plate, bowl, food) above the confidence threshold."""
        session = self._get_session()
        tensor, info = letterbox(image, self._size(session))
        out0, out1 = session.run(None, {"images": tensor})[:2]
        return decode(out0, out1, info, confidence=self.confidence, iou=self.iou)

    def segment(self, image: ImageBGR) -> tuple[list[FoodRegion], Mask | None]:
        """Return the food regions and the plate mask (None if no plate is found).

        The plate mask is what ``depth.plane.fit_plane`` uses as its reference. Models
        trained without a ``plate`` class (the Voxel51 baseline) always return None.
        """
        food_names = {c.value for c in FoodCategory}
        regions: list[FoodRegion] = []
        plate: Mask | None = None
        for det in self.detect(image):
            if det.class_name == "plate":
                if plate is None:  # detections are sorted, so the first is the best
                    plate = det.mask
            elif det.class_name in food_names:
                regions.append(FoodRegion(det.mask, FoodCategory(det.class_name), det.confidence))
        return regions, plate
