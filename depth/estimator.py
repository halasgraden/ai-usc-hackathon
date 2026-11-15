"""Monocular depth from Depth Anything 3 Small, and its conversion to centimetres."""

from __future__ import annotations

from core.types import DepthMap, ImageBGR, Mask


class DepthEstimator:
    """Wraps Depth Anything 3 Small. One forward pass per plate."""

    def predict(self, image: ImageBGR) -> DepthMap:
        """Relative depth for every pixel, same height and width as ``image``."""
        raise NotImplementedError("hour 4-5: load DA3 Small, resize output to the frame")


def to_centimetres(
    relative: DepthMap, plate: Mask, block: Mask, block_height_cm: float = 2.0
) -> DepthMap:
    """Rescale relative depth so the calibration block stands ``block_height_cm`` tall.

    The plate plane and the top of the 2 cm block give two known heights, which
    fixes the scale of a relative depth map for this frame. This assumes the
    model's output is linear in distance over the few centimetres that matter;
    ``eval/`` measures how well that holds (measuring-cup volume test).
    """
    raise NotImplementedError("hour 4-5")
