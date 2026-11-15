from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

from capture.scale import PLATE_DIAMETER_CM, PlateEllipse
from depth.estimator import (
    DepthEstimator,
    RawDepth,
    depth_scale,
    is_metric_checkpoint,
    load_da3,
    to_centimetres,
)
from depth.plane import fit_plane, height_map, volume_ml
from depth.volume import (
    ellipse_mask,
    food_volume_ml,
    measure_food_volumes,
    plate_distance_cm,
    volumes_from_depth,
)


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


# --- pinhole rendering of a plate with food --------------------------------

H, W, F = 480, 640, 600.0
"""Synthetic camera: 640 x 480 frame, focal length 600 px, principal point at the centre."""


def render(
    plate_z: float = 40.0,
    box: tuple[float, float, float, float, float] | None = (-3.0, 3.0, -2.0, 2.0, 2.5),
    cap: tuple[float, float, float, float] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], list[NDArray[np.bool_]]]:
    """Ray-cast a flat plate (diameter 21.59 cm, ``plate_z`` cm below the camera).

    ``box`` = (x0, x1, y0, y1, height) in cm on the plate, an axis-aligned block, or None.
    ``cap`` = (cx, cy, radius, height) in cm, a spherical cap (a mound of food).
    Returns true depth (z, in cm), the visible plate mask and the food masks.
    """
    v, u = np.indices((H, W), dtype=np.float64)
    rx, ry = (u + 0.5 - W / 2) / F, (v + 0.5 - H / 2) / F  # ray direction (rx, ry, 1)
    depth = np.full((H, W), plate_z + 3.0)  # tray, beyond the plate
    on_plate = (rx * plate_z) ** 2 + (ry * plate_z) ** 2 <= (PLATE_DIAMETER_CM / 2) ** 2
    depth[on_plate] = plate_z
    foods: list[NDArray[np.bool_]] = []

    box_mask = np.zeros((H, W), dtype=bool)
    if box is not None:
        x0, x1, y0, y1, h = box
        top = plate_z - h
        best = np.full((H, W), np.inf)
        hit_x, hit_y = rx * top, ry * top
        best[(hit_x >= x0) & (hit_x <= x1) & (hit_y >= y0) & (hit_y <= y1)] = top
        for wall, lo, hi, along in ((x0, y0, y1, rx), (x1, y0, y1, rx)):
            with np.errstate(divide="ignore", invalid="ignore"):
                z = wall / along
            y = ry * z
            ok = (z >= top) & (z <= plate_z) & (y >= lo) & (y <= hi)
            best = np.where(ok, np.minimum(best, z), best)
        for wall, lo, hi, along in ((y0, x0, x1, ry), (y1, x0, x1, ry)):
            with np.errstate(divide="ignore", invalid="ignore"):
                z = wall / along
            x = rx * z
            ok = (z >= top) & (z <= plate_z) & (x >= lo) & (x <= hi)
            best = np.where(ok, np.minimum(best, z), best)
        box_mask = np.isfinite(best)
        depth[box_mask] = best[box_mask]
        foods.append(box_mask)

    if cap is not None:
        cx, cy, a, hc = cap
        r = (a * a + hc * hc) / (2 * hc)  # sphere radius; centre at plate_z - hc + r
        cz = plate_z - hc + r
        # Ray p = t * (rx, ry, 1); solve |p - c|^2 = r^2 for the nearest t.
        bq = -2 * (rx * cx + ry * cy + cz)
        aq = rx * rx + ry * ry + 1
        cq = cx * cx + cy * cy + cz * cz - r * r
        disc = bq * bq - 4 * aq * cq
        with np.errstate(invalid="ignore"):
            t = (-bq - np.sqrt(disc)) / (2 * aq)
        cap_mask = (disc >= 0) & (t <= plate_z) & (t > 0) & ~box_mask
        depth[cap_mask] = t[cap_mask]
        foods.append(cap_mask)

    any_food = np.zeros((H, W), dtype=bool)
    for m in foods:
        any_food |= m
    return depth.astype(np.float32), on_plate & ~any_food, foods


def cap_volume(a: float, h: float) -> float:
    return float(np.pi * h * (3 * a * a + h * h) / 6)


def test_plate_distance_is_focal_times_scale() -> None:
    assert plate_distance_cm(0.0675, 600.0) == pytest.approx(40.5)
    with pytest.raises(ValueError):
        plate_distance_cm(0.0, 600.0)


def test_rendered_plate_scale_gives_back_the_camera_distance() -> None:
    depth, plate, foods = render()
    m = volumes_from_depth(depth, plate, foods, focal_px=F)
    assert m is not None
    # Plate spans 21.59 * 600 / 40 = 323.9 px; pixel quantisation only.
    assert m.cm_per_px == pytest.approx(40.0 / F, rel=0.005)
    assert m.plate_distance_cm == pytest.approx(40.0, rel=0.005)
    assert m.depth_scale == pytest.approx(1.0, rel=0.005)


@pytest.mark.parametrize("unknown_scale", [1.0, 0.0173, 3.7])
def test_food_volumes_recovered_from_up_to_scale_depth(unknown_scale: float) -> None:
    """A 6 x 4 x 2.5 cm block (60 ml) and a spherical-cap mound, depth known only up to scale."""
    cap = (5.0, 3.0, 2.5, 2.0)
    depth, plate, foods = render(cap=cap)
    m = volumes_from_depth(depth * np.float32(unknown_scale), plate, foods, focal_px=F)
    assert m is not None
    block_ml, mound_ml = m.volumes_ml
    # Tolerance 2% on noise-free synthetic depth: pixel quantisation of the
    # outlines and of the visible block wall. Says nothing about real DA3 depth.
    assert block_ml == pytest.approx(60.0, rel=0.02)
    assert mound_ml == pytest.approx(cap_volume(2.5, 2.0), rel=0.02)
    assert m.depth_scale * unknown_scale == pytest.approx(1.0, rel=0.005)


def test_perspective_footprint_matters_for_tall_food() -> None:
    depth, plate, foods = render(box=(-3.0, 3.0, -2.0, 2.0, 6.0))  # 144 ml block
    m = volumes_from_depth(depth, plate, foods, focal_px=F)
    assert m is not None
    assert m.volumes_ml[0] == pytest.approx(144.0, rel=0.02)
    # The plate's constant scale overestimates by about (40 / 34)^2 = 1.38.
    heights = height_map(depth, fit_plane(depth, plate))
    flat = volume_ml(heights, foods[0], cm_per_px=m.cm_per_px)
    assert flat > 1.3 * 144.0


def test_mound_volume_needs_the_slope_term() -> None:
    """A centred 2 cm mound: the flat (z/f)^2 footprint alone under-counts it."""
    depth, plate, foods = render(box=None, cap=(0.0, 0.0, 2.5, 2.0))
    true_ml = cap_volume(2.5, 2.0)
    heights = height_map(depth, fit_plane(depth, plate))
    mound = foods[0]
    assert food_volume_ml(heights, depth, mound, F) == pytest.approx(true_ml, rel=0.01)
    flat_only = float((heights * (depth.astype(np.float64) / F) ** 2)[mound].sum())
    assert flat_only < 0.97 * true_ml


def test_food_volume_matches_plane_volume_when_depth_is_constant() -> None:
    depth, _, block = scene()
    heights = height_map(depth, (0.0, 0.0, 40.0))
    const = np.full(depth.shape, 40.0, dtype=np.float32)
    assert food_volume_ml(heights, const, block, focal_px=400.0) == pytest.approx(
        volume_ml(heights, block, cm_per_px=0.1)
    )
    with pytest.raises(ValueError):
        food_volume_ml(heights, const, block, focal_px=0.0)
    with pytest.raises(ValueError):
        food_volume_ml(heights, const[:10], block, focal_px=400.0)


def test_rim_height_moves_the_plane_further_away() -> None:
    depth, plate, foods = render()
    flat = volumes_from_depth(depth, plate, foods, focal_px=F)
    rimmed = volumes_from_depth(depth, plate, foods, focal_px=F, rim_height_cm=1.5)
    assert flat is not None and rimmed is not None
    assert rimmed.plate_distance_cm == pytest.approx(flat.plate_distance_cm + 1.5)
    assert rimmed.volumes_ml[0] > flat.volumes_ml[0]


def test_clipped_plate_is_not_measured() -> None:
    depth, plate, foods = render(plate_z=25.0)  # plate wider than the frame
    assert volumes_from_depth(depth, plate, foods, focal_px=F) is None


def test_plate_fully_covered_by_food_is_not_measured() -> None:
    depth, plate, foods = render()
    # Food claims every plate pixel: no bare plate left to fit the plane to.
    everything = plate | foods[0]
    assert volumes_from_depth(depth, plate, [everything], focal_px=F) is None


def test_volumes_from_depth_rejects_bad_inputs() -> None:
    depth, plate, foods = render()
    with pytest.raises(ValueError):
        volumes_from_depth(depth[:10], plate, foods, focal_px=F)
    with pytest.raises(ValueError):
        volumes_from_depth(depth, plate, foods, focal_px=0.0)


def test_ellipse_mask_shrinks_about_the_centre() -> None:
    e = PlateEllipse(center=(50.0, 40.0), major_px=60.0, minor_px=60.0, angle_deg=0.0)
    full, inner = ellipse_mask((80, 100), e), ellipse_mask((80, 100), e, 0.5)
    assert full[40, 50] and full[40, 79] and not full[40, 81]
    assert inner[40, 64] and not inner[40, 66]
    assert full.sum() == pytest.approx(np.pi * 30**2, rel=0.02)


# --- depth scale ------------------------------------------------------------


def test_to_centimetres_puts_the_plate_at_the_given_distance() -> None:
    depth, plate, _ = scene(tilt=0.01)
    relative = depth * np.float32(0.02)
    cm = to_centimetres(relative, plate, 40.0, anchor_xy=(0.0, 0.0))
    np.testing.assert_allclose(cm, depth, rtol=1e-4)
    assert depth_scale(relative, plate, 40.0, anchor_xy=(0.0, 0.0)) == pytest.approx(50.0)
    # Default anchor: the plate centroid, where the true plane is not 40 cm.
    assert depth_scale(depth, plate, 40.0) != pytest.approx(1.0, abs=1e-3)


def test_depth_scale_rejects_bad_input() -> None:
    depth, plate, _ = scene()
    with pytest.raises(ValueError):
        depth_scale(depth, plate, 0.0)
    with pytest.raises(ValueError):
        depth_scale(-depth, plate, 40.0)


# --- estimator --------------------------------------------------------------


def fake_model(depth: NDArray[np.float32], focal: float | None) -> object:
    calls: list[NDArray[np.uint8]] = []

    def run(rgb: NDArray[np.uint8]) -> RawDepth:
        calls.append(rgb)
        return depth, focal

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_depth_estimator_returns_a_map_the_size_of_the_frame() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[..., 0] = 255  # blue in BGR
    model = fake_model(np.full((24, 32), 2.0, dtype=np.float32), focal=30.0)
    est = DepthEstimator(model)  # type: ignore[arg-type]
    pred = est.predict_full(frame)
    assert pred.depth.shape == (48, 64)
    assert pred.depth.dtype == np.float32
    np.testing.assert_allclose(pred.depth, 2.0)
    assert pred.focal_px == pytest.approx(60.0)  # half-resolution focal, doubled
    assert pred.metric is False
    rgb = model.calls[0]  # type: ignore[attr-defined]
    assert rgb[0, 0, 2] == 255 and rgb[0, 0, 0] == 0  # channels swapped to RGB
    assert est.predict(frame).shape == (48, 64)


def test_depth_estimator_accepts_batched_output_and_missing_focal() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    est = DepthEstimator(fake_model(np.ones((1, 48, 64), dtype=np.float32), None))  # type: ignore[arg-type]
    pred = est.predict_full(frame)
    assert pred.depth.shape == (48, 64) and pred.focal_px is None


@pytest.mark.parametrize("bad", [np.full((4, 4), np.nan), np.zeros((4, 4)), np.ones((2, 4, 4))])
def test_depth_estimator_rejects_bad_model_output(bad: NDArray[np.float64]) -> None:
    est = DepthEstimator(fake_model(bad.astype(np.float32), None))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        est.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        est.predict(np.zeros((8, 8), dtype=np.uint8))


def test_estimator_is_lazy_and_reports_missing_da3() -> None:
    est = DepthEstimator()  # must not import torch or load weights
    assert est.metric is False
    if importlib.util.find_spec("depth_anything_3") is not None:  # pragma: no cover
        pytest.skip("DA3 installed; the real loader is not exercised by unit tests")
    with pytest.raises(ImportError, match="Depth Anything 3"):
        est.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ImportError):
        load_da3()


def test_metric_checkpoint_names() -> None:
    assert is_metric_checkpoint("depth-anything/DA3METRIC-LARGE")
    assert is_metric_checkpoint("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    assert not is_metric_checkpoint("depth-anything/DA3-SMALL")
    assert not is_metric_checkpoint("depth-anything/DA3MONO-LARGE")
    assert DepthEstimator(model_id="depth-anything/DA3METRIC-LARGE").metric


def test_measure_food_volumes_end_to_end_with_an_injected_model() -> None:
    depth, plate, foods = render()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    est = DepthEstimator(fake_model(depth * np.float32(0.05), focal=F / 2))  # type: ignore[arg-type]
    # Focal from the model (rescaled from its own resolution) ...
    small = render()[0][::2, ::2] * np.float32(0.05)
    est_small = DepthEstimator(fake_model(small, focal=F / 2))  # type: ignore[arg-type]
    m = measure_food_volumes(frame, plate, foods, estimator=est_small)
    assert m is not None and m.focal_px == pytest.approx(F)
    # Downsampling the depth blurs edges, so a looser 5% tolerance here.
    assert m.volumes_ml[0] == pytest.approx(60.0, rel=0.05)
    # ... or from rig calibration, which wins over the model's guess.
    m2 = measure_food_volumes(frame, plate, foods, estimator=est, focal_px=F)
    assert m2 is not None and m2.volumes_ml[0] == pytest.approx(60.0, rel=0.02)
    no_focal = DepthEstimator(fake_model(depth, None))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="focal"):
        measure_food_volumes(frame, plate, foods, estimator=no_focal)
