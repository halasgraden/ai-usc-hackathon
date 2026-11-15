"""Geometry from a metric depth map: plate plane, food height, volume."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.types import DepthMap, Mask

Plane = tuple[float, float, float]
"""Coefficients ``(a, b, c)`` of the plate surface ``depth = a*x + b*y + c`` in pixels."""


def fit_plane(depth_cm: DepthMap, plate: Mask) -> Plane:
    """Least-squares plane through the bare-plate pixels.

    Args:
        depth_cm: distance from the camera in centimetres.
        plate: True on visible plate surface (not under food).

    Raises:
        ValueError: fewer than 3 plate pixels, or shapes differ.
    """
    if depth_cm.shape != plate.shape:
        raise ValueError("depth map and plate mask must have the same shape")
    ys, xs = np.nonzero(plate)
    if xs.size < 3:
        raise ValueError("need at least 3 plate pixels to fit a plane")
    design = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    coeffs, *_ = np.linalg.lstsq(design, depth_cm[ys, xs].astype(np.float64), rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def fit_plane_robust(depth_cm: DepthMap, plate: Mask, keep: float = 0.8, rounds: int = 2) -> Plane:
    """``fit_plane``, refit ``rounds`` times on the ``keep`` fraction of pixels closest to it.

    Real depth (Depth Anything 3) puts false height on bare plate next to food (edge blur) and
    on the raised rim; a plain least-squares fit is pulled toward those pixels, which lifts the
    plane and under-counts food. Trimming the worst-fitting pixels removes that pull. On a
    perfectly flat plate it returns the same plane as ``fit_plane``.

    Raises:
        ValueError: as ``fit_plane``, or ``keep`` outside (0, 1].
    """
    if not 0.0 < keep <= 1.0:
        raise ValueError("keep must be in (0, 1]")
    plane = fit_plane(depth_cm, plate)
    ys, xs = np.nonzero(plate)
    values = depth_cm[ys, xs].astype(np.float64)
    for _ in range(rounds):
        a, b, c = plane
        residual = np.abs(a * xs + b * ys + c - values)
        cut = float(np.quantile(residual, keep))
        inliers = residual <= cut
        if int(inliers.sum()) < 3:
            break
        sub = np.zeros(plate.shape, dtype=bool)
        sub[ys[inliers], xs[inliers]] = True
        plane = fit_plane(depth_cm, sub)
    return plane


def height_map(depth_cm: DepthMap, plane: Plane) -> NDArray[np.float64]:
    """Height of every pixel above the plate plane, in cm; below-plane values clip to 0.

    Food is closer to the camera than the plate, so height = plane depth - depth.
    """
    a, b, c = plane
    ys, xs = np.indices(depth_cm.shape)
    return np.clip(a * xs + b * ys + c - depth_cm.astype(np.float64), 0.0, None)


def volume_ml(height_cm: NDArray[np.float64], region: Mask, cm_per_px: float) -> float:
    """Volume under a food mask: sum of pixel height x pixel area. 1 cm^3 = 1 ml.

    Raises:
        ValueError: non-positive scale, or shapes differ.
    """
    if height_cm.shape != region.shape:
        raise ValueError("height map and region mask must have the same shape")
    if cm_per_px <= 0:
        raise ValueError("cm_per_px must be positive")
    return float(height_cm[region].sum() * cm_per_px**2)
