"""Pixel-to-centimetre scale for a frame, taken from the plate itself.

Every plate at the USC Village dish return is the same 8.5 inch (21.59 cm)
plate, so the plate's outline in the image is a ruler of known length. The
plate mask comes from the segmentation model ("plate" class). We fill its
convex hull (food covers part of the plate, so the raw mask has holes and bites
out of the edge), fit an ellipse to the hull, and divide the plate diameter by
the ellipse's major axis. An ellipse rather than a circle: a camera that is not
exactly overhead sees the round rim as an ellipse whose major axis still spans
the full diameter (to first order), while the minor axis is foreshortened.

The scale applies at the height of the plate rim. See ``docs/MEASUREMENT.md``
for how the depth stage uses it and for the failure modes.

The ArUco path (``detect_scale``) is kept because it is a few lines with
``cv2.aruco`` and is handy for a bench calibration of the camera's focal
length, but the rig does not rely on a marker.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.types import ImageBGR, Mask, ScaleCalibration

PLATE_DIAMETER_CM = 21.59
"""Outer diameter of a USC Village dining plate: 8.5 inches."""

MARKER_SIZE_CM = 4.0
"""Side of the black square of the ArUco marker used for bench calibration."""

MIN_PLATE_AREA_PX = 400
"""Smaller plate masks are noise, not a plate."""

MIN_AXIS_RATIO = 0.8
"""Minor/major below this means a tilt beyond ~37 degrees, or not a round dish."""

FILL_RATIO_RANGE = (0.95, 1.05)
"""Allowed (filled hull area) / (fitted ellipse area). A square scores about 0.93."""


@dataclass(frozen=True, slots=True)
class PlateEllipse:
    """Ellipse fitted to the plate outline, in pixels.

    Attributes:
        center: ``(x, y)`` of the ellipse centre.
        major_px: full length of the major axis.
        minor_px: full length of the minor axis.
        angle_deg: rotation as reported by ``cv2.fitEllipse``.
    """

    center: tuple[float, float]
    major_px: float
    minor_px: float
    angle_deg: float


def touches_border(mask: Mask, margin_px: int = 2) -> bool:
    """True when any mask pixel lies within ``margin_px`` of the image edge.

    Raises:
        ValueError: the mask is not 2-D.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    m = max(1, margin_px)
    return bool(mask[:m].any() or mask[-m:].any() or mask[:, :m].any() or mask[:, -m:].any())


def fit_plate_ellipse(plate: Mask, border_margin_px: int = 2) -> PlateEllipse | None:
    """Fit an ellipse to the convex hull of the largest plate blob.

    Returns None when the outline cannot be trusted as a whole plate rim: too
    small, touching the image border (plate partly out of frame, so the hull is
    cut by the edge), too elongated, or not elliptical (a tray, a square dish,
    or a mask that merged the plate with something next to it).

    Raises:
        ValueError: the mask is not 2-D.
    """
    if plate.ndim != 2:
        raise ValueError("plate mask must be 2-D")
    if int(plate.sum()) < MIN_PLATE_AREA_PX or touches_border(plate, border_margin_px):
        return None
    contours, _ = cv2.findContours(plate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    filled = np.zeros(plate.shape, dtype=np.uint8)
    cv2.fillPoly(filled, [hull], 1)
    hull_area = float(filled.sum())
    if hull_area < MIN_PLATE_AREA_PX:
        return None
    # Fit to the dense outline of the filled hull: hull vertices alone are few
    # and unevenly spaced. The outline runs through boundary pixel centres, half
    # a pixel inside the region's edge on each side, hence the +1 px.
    outline, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(outline, key=len)
    if len(points) < 5:
        return None
    (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(points)
    major, minor = float(max(ax1, ax2)) + 1.0, float(min(ax1, ax2)) + 1.0
    if major <= 0 or minor / major < MIN_AXIS_RATIO:
        return None
    fill = hull_area / (np.pi * major * minor / 4.0)
    if not FILL_RATIO_RANGE[0] <= fill <= FILL_RATIO_RANGE[1]:
        return None
    return PlateEllipse((float(cx), float(cy)), major, minor, float(angle))


def scale_from_ellipse(
    ellipse: PlateEllipse, plate_diameter_cm: float = PLATE_DIAMETER_CM
) -> ScaleCalibration:
    """cm per pixel at rim height: the known diameter over the major axis length.

    Raises:
        ValueError: non-positive diameter or axis.
    """
    if plate_diameter_cm <= 0 or ellipse.major_px <= 0:
        raise ValueError("plate diameter and major axis must be positive")
    return ScaleCalibration(cm_per_px=plate_diameter_cm / ellipse.major_px)


def scale_from_plate(
    plate: Mask, plate_diameter_cm: float = PLATE_DIAMETER_CM, border_margin_px: int = 2
) -> ScaleCalibration | None:
    """Scale of the frame from the plate mask, or None if the plate is unusable.

    None means the frame must not be measured: the plate is clipped by the
    image border, too small, or its outline is not a plausible round plate.
    A non-standard dish that is also round (a bowl, a side plate) is NOT
    detected here and would give a wrong scale; that has to be caught upstream
    by the segmentation class ("bowl" vs "plate").
    """
    ellipse = fit_plate_ellipse(plate, border_margin_px)
    if ellipse is None:
        return None
    return scale_from_ellipse(ellipse, plate_diameter_cm)


def detect_scale(
    image: ImageBGR, marker_size_cm: float = MARKER_SIZE_CM
) -> ScaleCalibration | None:
    """Find an ArUco marker (DICT_4X4_50) and derive the scale at its plane.

    Uses the mean of the four side lengths of the first marker found. Returns
    None when no marker is visible. Not used on the rig (the plate is the scale
    reference, see ``scale_from_plate``); kept for bench calibration.

    Raises:
        ValueError: non-positive marker size, or an image that is not H x W x 3.
    """
    if marker_size_cm <= 0:
        raise ValueError("marker size must be positive")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be H x W x 3 BGR")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or len(corners) == 0:
        return None
    quad = np.asarray(corners[0], dtype=np.float64).reshape(4, 2)
    mean_side = float(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1).mean())
    if mean_side <= 0:
        return None
    return ScaleCalibration(cm_per_px=marker_size_cm / mean_side)
