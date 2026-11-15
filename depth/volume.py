"""End-to-end measure stage: frame + plate mask + food masks -> volume in ml per food.

Pipeline (see ``docs/MEASUREMENT.md``):

1. Plate ellipse -> cm per pixel at rim height (``capture.scale``).
2. Plate distance from the camera: ``Z = focal_px * cm_per_px`` (pinhole model),
   plus the rim height to reach the well where food sits.
3. Depth model -> up-to-scale depth; scale it so the plate plane, fitted to
   bare plate pixels in the well, lies at ``Z`` (``depth.estimator.to_centimetres``).
4. Height above that plane (``depth.plane``), then volume per food mask,
   integrating each pixel's true ground footprint under perspective
   (``food_volume_ml``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from capture.scale import PLATE_DIAMETER_CM, PlateEllipse, fit_plate_ellipse, scale_from_ellipse
from core.types import DepthMap, ImageBGR, Mask
from depth.estimator import DepthEstimator, depth_scale
from depth.plane import fit_plane, height_map

WELL_FRACTION = 0.75
"""Plane-fit pixels are kept inside this fraction of the rim ellipse, to skip the
sloped rim. A guess from typical plate shapes; not measured on a Village plate."""

MIN_SURFACE_PX = 50
"""Below this many bare well pixels the plane fit falls back to all bare plate pixels."""


@dataclass(frozen=True, slots=True)
class PlateMeasurement:
    """Volumes for one plate plus the numbers that produced them (for logging/QA).

    Attributes:
        volumes_ml: one volume per food mask, in the order given.
        cm_per_px: scale at the plate rim, from the plate diameter.
        focal_px: focal length used, in frame pixels.
        plate_distance_cm: camera-to-plate-well distance used to fix the depth scale.
        depth_scale: factor applied to the model's depth. Near 1 for a metric model
            whose output agrees with the plate; arbitrary for a relative model.
        ellipse: the fitted plate rim.
    """

    volumes_ml: tuple[float, ...]
    cm_per_px: float
    focal_px: float
    plate_distance_cm: float
    depth_scale: float
    ellipse: PlateEllipse


def plate_distance_cm(cm_per_px: float, focal_px: float) -> float:
    """Pinhole camera: an object ``Z`` cm away spans ``Z / f`` cm per pixel, so ``Z = f * s``.

    Raises:
        ValueError: non-positive input.
    """
    if cm_per_px <= 0 or focal_px <= 0:
        raise ValueError("cm_per_px and focal_px must be positive")
    return focal_px * cm_per_px


def _gradient_within(z: NDArray[np.float64], region: Mask, axis: int) -> NDArray[np.float64]:
    """Finite-difference gradient along ``axis`` using only pairs of pixels inside ``region``.

    Differences across the region border (food edge to plate, a depth jump at
    an occlusion) are skipped; a pixel with no in-region neighbour gets 0.
    """
    zz = np.moveaxis(z, axis, 1)
    rr = np.moveaxis(region, axis, 1)
    diff = zz[:, 1:] - zz[:, :-1]
    ok = rr[:, 1:] & rr[:, :-1]
    total = np.zeros_like(zz)
    count = np.zeros_like(zz)
    total[:, 1:] += np.where(ok, diff, 0.0)
    total[:, :-1] += np.where(ok, diff, 0.0)
    count[:, 1:] += ok
    count[:, :-1] += ok
    grad = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return np.moveaxis(grad, 1, axis)


def food_volume_ml(
    height_cm: NDArray[np.float64],
    depth_cm: DepthMap,
    region: Mask,
    focal_px: float,
    principal_xy: tuple[float, float] | None = None,
) -> float:
    """Volume under a mask, integrating height over the true ground area of each pixel.

    A pixel at ``(u, v)`` sees the surface point ``(X, Y) = z * (r_x, r_y)`` with
    ``r = ((u, v) - principal point) / f``. The ground area it covers is the
    Jacobian of that map: ``(z/f)^2 + (z/f) * (r_x dz/du + r_y dz/dv)``. The
    first term alone (a flat footprint at the pixel's depth) under-counts a
    mound whose sides face away from the optical axis; the second term fixes
    that and is zero on a vertical wall seen edge-on. Using the plate's scale
    for every pixel (``depth.plane.volume_ml``) would instead over-count tall
    food by roughly ``(Z_plate / z_food)^2``. Gradients are taken only between
    pixels inside ``region``; negative areas (folds at occlusions) clip to 0.

    Args:
        height_cm: height above the plate plane.
        depth_cm: metric depth along the optical axis.
        region: the food mask.
        focal_px: focal length in pixels.
        principal_xy: principal point in pixel-index coordinates; default the
            image centre ``((W - 1) / 2, (H - 1) / 2)``.

    Raises:
        ValueError: shapes differ or non-positive focal length.
    """
    if not (height_cm.shape == depth_cm.shape == region.shape):
        raise ValueError("height map, depth map and region mask must have the same shape")
    if focal_px <= 0:
        raise ValueError("focal_px must be positive")
    rows, cols = depth_cm.shape
    cx, cy = principal_xy if principal_xy is not None else ((cols - 1) / 2, (rows - 1) / 2)
    z = depth_cm.astype(np.float64)
    v, u = np.indices(z.shape, dtype=np.float64)
    rx, ry = (u - cx) / focal_px, (v - cy) / focal_px
    slope = rx * _gradient_within(z, region, 1) + ry * _gradient_within(z, region, 0)
    area = np.clip((z / focal_px) ** 2 + (z / focal_px) * slope, 0.0, None)
    return float((height_cm * area)[region].sum())


def ellipse_mask(shape: tuple[int, int], ellipse: PlateEllipse, fraction: float = 1.0) -> Mask:
    """Pixels inside the ellipse shrunk about its centre by ``fraction``."""
    ys, xs = np.indices(shape, dtype=np.float64)
    cx, cy = ellipse.center
    theta = np.deg2rad(ellipse.angle_deg)
    dx, dy = xs - cx, ys - cy
    # cv2.fitEllipse: the first axis lies along ``angle``; which one is major is
    # not fixed, so test against both orientations' semi-axes via the larger/smaller.
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    a = 0.5 * fraction * ellipse.major_px
    b = 0.5 * fraction * ellipse.minor_px
    inside_ab = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    inside_ba = (u / b) ** 2 + (v / a) ** 2 <= 1.0
    # For a near-circular plate the two agree; take the intersection to stay
    # conservative (only well pixels) whichever way the axes are labelled.
    mask: Mask = inside_ab & inside_ba
    return mask


def volumes_from_depth(
    depth: DepthMap,
    plate_mask: Mask,
    food_masks: Sequence[Mask],
    *,
    focal_px: float,
    plate_diameter_cm: float = PLATE_DIAMETER_CM,
    rim_height_cm: float = 0.0,
    well_fraction: float = WELL_FRACTION,
) -> PlateMeasurement | None:
    """Per-food volume from an up-to-scale (or metric) depth map of one plate.

    Args:
        depth: H x W depth from the model, any units proportional to distance.
        plate_mask: visible plate pixels (may exclude food; holes are fine).
        food_masks: one mask per food region.
        focal_px: camera focal length in frame pixels.
        plate_diameter_cm: rim diameter of the plate.
        rim_height_cm: height of the rim above the well floor. 0 treats the plate
            as flat; the Village plate's value has not been measured yet.
        well_fraction: see ``WELL_FRACTION``.

    Returns:
        None when the plate cannot serve as the scale (clipped, implausible) or
        no bare plate is visible to fit the plane.

    Raises:
        ValueError: mismatched shapes or non-positive focal length.
    """
    if depth.shape != plate_mask.shape or any(m.shape != depth.shape for m in food_masks):
        raise ValueError("depth map and all masks must have the same shape")
    if focal_px <= 0:
        raise ValueError("focal_px must be positive")
    ellipse = fit_plate_ellipse(plate_mask)
    if ellipse is None:
        return None
    cm_per_px = scale_from_ellipse(ellipse, plate_diameter_cm).cm_per_px
    distance = plate_distance_cm(cm_per_px, focal_px) + rim_height_cm

    food_any = np.zeros(depth.shape, dtype=bool)
    for m in food_masks:
        food_any |= m
    bare = plate_mask & ~food_any
    surface = bare & ellipse_mask(depth.shape, ellipse, well_fraction)
    if int(surface.sum()) < MIN_SURFACE_PX:
        surface = bare
    if int(surface.sum()) < 3:
        return None

    scale = depth_scale(depth, surface, distance, anchor_xy=ellipse.center)
    depth_cm = (depth.astype(np.float64) * scale).astype(np.float32)
    heights = height_map(depth_cm, fit_plane(depth_cm, surface))
    volumes = tuple(food_volume_ml(heights, depth_cm, m, focal_px) for m in food_masks)
    return PlateMeasurement(volumes, cm_per_px, focal_px, distance, scale, ellipse)


def measure_food_volumes(
    image: ImageBGR,
    plate_mask: Mask,
    food_masks: Sequence[Mask],
    *,
    estimator: DepthEstimator,
    focal_px: float | None = None,
    plate_diameter_cm: float = PLATE_DIAMETER_CM,
    rim_height_cm: float = 0.0,
) -> PlateMeasurement | None:
    """The measure stage: one frame and its masks in, volume in ml per food out.

    ``focal_px`` should be the rig camera's calibrated focal length in frame
    pixels. If None, the depth model's predicted focal length is used.

    Returns:
        None when the frame must not be measured (see ``volumes_from_depth``).

    Raises:
        ValueError: no focal length available, or bad shapes.
    """
    prediction = estimator.predict_full(image)
    focal = focal_px if focal_px is not None else prediction.focal_px
    if focal is None:
        raise ValueError("no focal length: pass focal_px or use a model that predicts intrinsics")
    return volumes_from_depth(
        prediction.depth,
        plate_mask,
        food_masks,
        focal_px=focal,
        plate_diameter_cm=plate_diameter_cm,
        rim_height_cm=rim_height_cm,
    )
