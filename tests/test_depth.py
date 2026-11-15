from __future__ import annotations

import numpy as np
import pytest

from depth.estimator import DepthEstimator
from depth.plane import fit_plane, height_map, volume_ml


def scene(tilt: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A plate 40 cm from the camera with a 2 cm tall, 50 x 50 px block on it."""
    ys, xs = np.indices((120, 160))
    depth = (40.0 + tilt * xs + 0.5 * tilt * ys).astype(np.float32)
    block = np.zeros(depth.shape, dtype=bool)
    block[30:80, 50:100] = True
    depth[block] -= 2.0
    return depth, ~block, block


@pytest.mark.parametrize("tilt", [0.0, 0.01])
def test_known_block_volume_is_recovered(tilt: float) -> None:
    depth, plate, block = scene(tilt)
    heights = height_map(depth, fit_plane(depth, plate))
    # 50 x 50 px at 0.1 cm/px = 5 x 5 cm footprint, 2 cm tall = 50 ml.
    assert volume_ml(heights, block, cm_per_px=0.1) == pytest.approx(50.0, rel=1e-3)
    assert volume_ml(heights, plate, cm_per_px=0.1) == pytest.approx(0.0, abs=1e-3)


def test_pixels_below_the_plane_clip_to_zero() -> None:
    depth, plate, _ = scene()
    depth[0, 0] += 5.0  # a dent, e.g. the gap beyond the plate rim
    assert height_map(depth, (0.0, 0.0, 40.0))[0, 0] == 0.0
    assert plate[0, 0]


def test_bad_inputs() -> None:
    depth, plate, block = scene()
    with pytest.raises(ValueError):
        fit_plane(depth, np.zeros_like(plate))
    with pytest.raises(ValueError):
        fit_plane(depth, plate[:10])
    with pytest.raises(ValueError):
        volume_ml(height_map(depth, (0.0, 0.0, 40.0)), block, cm_per_px=0.0)


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="depth model: hour 4-5")
def test_depth_estimator_returns_a_map_the_size_of_the_frame() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert DepthEstimator().predict(frame).shape == (48, 64)
