"""Pixel-to-centimetre scale from the ArUco marker fixed to the rig."""

from __future__ import annotations

from core.types import ImageBGR, ScaleCalibration

MARKER_SIZE_CM = 4.0


def detect_scale(
    image: ImageBGR, marker_size_cm: float = MARKER_SIZE_CM
) -> ScaleCalibration | None:
    """Find the rig's ArUco marker and derive the scale at the tray plane.

    Returns None when the marker is not visible; such a frame must not be
    measured, because phone lens and zoom vary and no stored scale is trusted.
    """
    raise NotImplementedError("hour 2-3: cv2.aruco.ArucoDetector, mean side length in px")
