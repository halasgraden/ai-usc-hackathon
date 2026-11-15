from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from capture.face_guard import contains_face
from capture.scale import detect_scale
from capture.source import parse_source
from capture.trigger import PlateTrigger, sharpness


def test_parse_source() -> None:
    assert parse_source(" 1 ") == 1
    assert parse_source("http://10.0.0.5:8080/video") == "http://10.0.0.5:8080/video"
    with pytest.raises(ValueError):
        parse_source("  ")


def test_sharp_image_scores_higher_than_its_blur() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, size=(64, 64)).astype(np.uint8)
    blurred = (
        sharp[:-1, :-1].astype(np.float32) + sharp[1:, :-1] + sharp[:-1, 1:] + sharp[1:, 1:]
    ) / 4
    assert sharpness(sharp) > sharpness(blurred.astype(np.float32)) > 0.0
    assert sharpness(np.full((10, 10), 128, dtype=np.uint8)) == 0.0


def test_sharpness_rejects_colour_or_tiny_images() -> None:
    with pytest.raises(ValueError):
        sharpness(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        sharpness(np.zeros((2, 8), dtype=np.uint8))


FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="trigger: hour 2-3")
def test_a_still_blank_scene_never_triggers() -> None:
    assert PlateTrigger().update(FRAME, datetime(2026, 11, 14, 18, 0)) is None


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="ArUco scale: hour 2-3")
def test_no_marker_means_no_scale() -> None:
    assert detect_scale(FRAME) is None


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="face guard: hour 6+")
def test_blank_frame_has_no_face() -> None:
    assert contains_face(FRAME) is False
