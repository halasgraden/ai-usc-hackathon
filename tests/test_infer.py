"""Tests for ``vision.infer`` using synthetic network output (no model file needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from core.types import SEG_CLASSES, FoodCategory
from vision.infer import Detection, LetterboxInfo, Segmenter, decode, letterbox

SIZE = 640
N_CLS = len(SEG_CLASSES)
N_PROTO = 32
N_ANCHORS = 100


def _blank_outputs() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    out0 = np.zeros((1, 4 + N_CLS + N_PROTO, N_ANCHORS), dtype=np.float32)
    protos = np.full((1, N_PROTO, SIZE // 4, SIZE // 4), -20.0, dtype=np.float32)
    return out0, protos


def _put(
    out0: NDArray[np.float32],
    anchor: int,
    box_cxcywh: tuple[float, float, float, float],
    class_name: str,
    score: float,
) -> None:
    out0[0, 0:4, anchor] = box_cxcywh
    out0[0, 4 + SEG_CLASSES.index(class_name), anchor] = score
    out0[0, 4 + N_CLS, anchor] = 1.0  # mask coefficient 0 -> uses prototype 0


def _square_info() -> LetterboxInfo:
    return LetterboxInfo(scale=1.0, pad_x=0, pad_y=0, orig_h=SIZE, orig_w=SIZE)


def test_letterbox_pads_wide_image_and_records_geometry() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    tensor, info = letterbox(image, SIZE)
    assert tensor.shape == (1, 3, SIZE, SIZE) and tensor.dtype == np.float32
    assert info.scale == pytest.approx(3.2)
    assert (info.pad_x, info.pad_y) == (0, 160)
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)  # padding row
    assert tensor[0, 0, 320, 320] == pytest.approx(1.0)  # image centre


def test_letterbox_converts_bgr_to_rgb() -> None:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    tensor, _ = letterbox(image, SIZE)
    assert tensor[0, 2].mean() == pytest.approx(1.0) and tensor[0, 0].mean() == 0.0


def test_decode_returns_nothing_below_threshold() -> None:
    out0, protos = _blank_outputs()
    _put(out0, 0, (320, 320, 100, 100), "bread", 0.1)
    assert decode(out0, protos, _square_info(), confidence=0.25) == []


def test_decode_places_box_and_mask_in_original_coordinates() -> None:
    out0, protos = _blank_outputs()
    protos[0, 0, 40:80, 40:80] = 20.0  # prototype blob = letterbox px 160..320
    _put(out0, 3, (240, 240, 160, 160), "cut_fruit", 0.9)
    (det,) = decode(out0, protos, _square_info())
    assert det.class_name == "cut_fruit" and det.confidence == pytest.approx(0.9)
    assert det.box_xyxy == pytest.approx((160, 160, 320, 320))
    ys, xs = np.nonzero(det.mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == pytest.approx((160, 319, 160, 319), abs=2)


def test_decode_undoes_letterbox_padding_and_scale() -> None:
    # 320x640 original (h x w): scale 1.0, vertical pad 160.
    info = LetterboxInfo(scale=1.0, pad_x=0, pad_y=160, orig_h=320, orig_w=640)
    out0, protos = _blank_outputs()
    protos[0, 0, 60:100, 60:100] = 20.0  # letterbox px 240..400 -> original y 80..240
    _put(out0, 0, (320, 320, 160, 160), "poultry", 0.8)
    (det,) = decode(out0, protos, info)
    assert det.mask.shape == (320, 640)
    assert det.box_xyxy == pytest.approx((240, 80, 400, 240))
    ys, xs = np.nonzero(det.mask)
    assert (xs.min(), ys.min()) == pytest.approx((240, 80), abs=2)


def test_nms_merges_duplicates_but_keeps_other_classes() -> None:
    out0, protos = _blank_outputs()
    _put(out0, 0, (300, 300, 100, 100), "bread", 0.9)
    _put(out0, 1, (302, 301, 100, 100), "bread", 0.6)  # duplicate
    _put(out0, 2, (300, 300, 100, 100), "eggs", 0.7)  # same box, different class
    dets = decode(out0, protos, _square_info())
    assert [d.class_name for d in dets] == ["bread", "eggs"]


def test_decode_rejects_mismatched_class_count() -> None:
    out0, protos = _blank_outputs()
    with pytest.raises(ValueError, match="channels"):
        decode(out0, protos, _square_info(), class_names=SEG_CLASSES[:5])


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, out0: NDArray[np.float32], out1: NDArray[np.float32]) -> None:
        self._outs = [out0, out1]
        self.seen_shape: tuple[int, ...] | None = None

    def get_inputs(self) -> list[Any]:
        class _In:
            def __init__(self) -> None:
                self.shape = [1, 3, SIZE, SIZE]

        return [_In()]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[NDArray[np.float32]]:
        self.seen_shape = input_feed["images"].shape
        return self._outs


def test_segmenter_splits_plate_from_food_and_skips_bowls() -> None:
    out0, protos = _blank_outputs()
    protos[0, 0, 20:140, 20:140] = 20.0
    _put(out0, 0, (320, 320, 500, 500), "plate", 0.95)
    _put(out0, 1, (300, 300, 120, 120), "rice_grains", 0.8)
    _put(out0, 2, (500, 500, 60, 60), "bowl", 0.7)
    session = _FakeSession(out0, protos)
    seg = Segmenter(Path("unused.onnx"), session=session)
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    regions, plate = seg.segment(image)
    assert session.seen_shape == (1, 3, SIZE, SIZE)
    assert plate is not None and plate.shape == (SIZE, SIZE)
    assert [r.category for r in regions] == [FoodCategory.RICE_GRAINS]
    assert regions[0].confidence == pytest.approx(0.8)
    assert all(isinstance(d, Detection) for d in seg.detect(image))


def test_segmenter_without_plate_class_returns_none_for_plate() -> None:
    out0, protos = _blank_outputs()
    _put(out0, 0, (300, 300, 120, 120), "eggs", 0.8)
    seg = Segmenter(Path("unused.onnx"), session=_FakeSession(out0, protos))
    regions, plate = seg.segment(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    assert plate is None and len(regions) == 1
