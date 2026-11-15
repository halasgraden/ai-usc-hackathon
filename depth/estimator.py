"""Monocular depth from Depth Anything 3, and its conversion to centimetres.

The model is behind a plain callable (``DepthModel``) so the rest of the
pipeline and the tests never need weights. ``load_da3`` builds that callable
from the real ``depth_anything_3`` package, imported lazily.

NOT YET RUN: ``load_da3`` is written against the DA3 README and
``depth_anything_3/api.py`` (``DepthAnything3.from_pretrained(...)``,
``model.inference([image], process_res=...)`` returning a prediction with
``depth`` [N, H, W] and ``intrinsics`` [N, 3, 3] at the processed resolution).
The sandbox it was written in cannot reach Hugging Face or GitHub, so it has
never loaded weights or produced a depth map. Install DA3 from the official
ByteDance-Seed GitHub repository; the ``depth-anything-3`` package on PyPI is
not published by the authors.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from core.types import DepthMap, ImageBGR, Mask
from depth.plane import fit_plane

DA3_SMALL = "depth-anything/DA3-SMALL"
"""Default checkpoint: relative (up-to-scale) depth plus predicted intrinsics."""

DA3_METRIC_LARGE = "depth-anything/DA3METRIC-LARGE"
"""Metric checkpoint; output becomes metres via ``focal_px * output / 300``."""

DA3_PROCESS_RES = 504
"""DA3's default processing resolution (longest side, ``upper_bound_resize``)."""

RawDepth = tuple[NDArray[Any], float | None]
"""Model output: depth at the model's own resolution, and focal length in pixels
of that same resolution (None if the model does not predict intrinsics)."""

DepthModel = Callable[[NDArray[np.uint8]], RawDepth]
"""Takes an H x W x 3 RGB uint8 image, returns ``RawDepth``."""


@dataclass(frozen=True, slots=True)
class DepthPrediction:
    """Depth resized to the frame, with the focal length rescaled to frame pixels.

    Attributes:
        depth: H x W depth (distance along the optical axis). In centimetres if
            ``metric``, otherwise in arbitrary units (correct up to one scale factor).
        focal_px: focal length in frame pixels as predicted by the model, or None.
        metric: whether ``depth`` claims to be in centimetres.
    """

    depth: DepthMap
    focal_px: float | None
    metric: bool


class DepthEstimator:
    """Depth Anything 3 for one frame at a time. One forward pass per plate.

    Args:
        model: a ``DepthModel`` callable. If None, ``load_da3(model_id, ...)`` is
            called on the first prediction (lazy: importing this module or
            constructing the estimator never imports torch).
        model_id: Hugging Face id of the DA3 checkpoint used when ``model`` is None.
        metric: whether the model's output is metric (in centimetres).
        device: torch device string for the lazy loader; None picks CUDA if present.
    """

    def __init__(
        self,
        model: DepthModel | None = None,
        *,
        model_id: str = DA3_SMALL,
        metric: bool | None = None,
        device: str | None = None,
    ) -> None:
        self._model = model
        self.model_id = model_id
        self.metric = is_metric_checkpoint(model_id) if metric is None else metric
        self.device = device

    def _get_model(self) -> DepthModel:
        if self._model is None:
            self._model = load_da3(self.model_id, device=self.device)
        return self._model

    def predict_full(self, image: ImageBGR) -> DepthPrediction:
        """Depth and focal length for ``image``, both at the frame's resolution.

        Raises:
            ValueError: the image is not H x W x 3, or the model output is not a
                finite, positive 2-D map.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be H x W x 3 BGR")
        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        raw, focal = self._get_model()(rgb)
        raw = np.squeeze(np.asarray(raw, dtype=np.float32))
        if raw.ndim != 2:
            raise ValueError(f"depth model returned shape {raw.shape}, expected 2-D")
        if not np.isfinite(raw).all() or (raw <= 0).any():
            raise ValueError("depth model returned non-finite or non-positive depth")
        mh, mw = raw.shape
        if (mh, mw) != (height, width):
            depth = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            depth = raw
        frame_focal = None
        if focal is not None:
            frame_focal = float(focal) * 0.5 * (width / mw + height / mh)
        return DepthPrediction(np.asarray(depth, dtype=np.float32), frame_focal, self.metric)

    def predict(self, image: ImageBGR) -> DepthMap:
        """Depth for every pixel, same height and width as ``image``."""
        return self.predict_full(image).depth


def is_metric_checkpoint(model_id: str) -> bool:
    """DA3METRIC-* and DA3NESTED-* output metric depth; the main and mono series do not."""
    name = model_id.rsplit("/", 1)[-1].upper()
    return name.startswith(("DA3METRIC", "DA3NESTED"))


def load_da3(
    model_id: str = DA3_SMALL,
    *,
    device: str | None = None,
    process_res: int = DA3_PROCESS_RES,
    focal_px_hint: float | None = None,
) -> DepthModel:
    """Load a Depth Anything 3 checkpoint and wrap it as a ``DepthModel``.

    NOT YET RUN (no network to the weights in the dev sandbox); see the module
    docstring. Metric checkpoints are converted to centimetres: DA3METRIC
    output is ``focal * output / 300`` metres per the DA3 README, with the focal
    length in processed-image pixels; DA3NESTED output is already in metres.

    Args:
        model_id: Hugging Face checkpoint id, e.g. ``DA3_SMALL``.
        device: torch device; None picks ``cuda`` when available, else ``cpu``.
        process_res: DA3's processing resolution.
        focal_px_hint: focal length in pixels of the processed image, used for a
            DA3METRIC checkpoint when the prediction carries no intrinsics.

    Raises:
        ImportError: ``depth_anything_3`` or torch is not installed.
    """
    try:
        torch = importlib.import_module("torch")
        da3_api = importlib.import_module("depth_anything_3.api")
        pil_image = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 is not installed: pip install torch torchvision, then "
            "install the ByteDance-Seed/Depth-Anything-3 repository (see docs/MEASUREMENT.md)"
        ) from exc
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = da3_api.DepthAnything3.from_pretrained(model_id).to(device=dev)
    net.eval()
    name = model_id.rsplit("/", 1)[-1].upper()

    def run(rgb: NDArray[np.uint8]) -> RawDepth:
        # A PIL image makes the channel order unambiguous (RGB).
        with torch.inference_mode():
            pred = net.inference([pil_image.fromarray(rgb)], process_res=process_res)
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        focal: float | None = None
        intrinsics = getattr(pred, "intrinsics", None)
        if intrinsics is not None:
            k = np.asarray(intrinsics[0], dtype=np.float64)
            focal = float(0.5 * (k[0, 0] + k[1, 1]))
        if name.startswith("DA3METRIC"):
            f = focal if focal is not None else focal_px_hint
            if f is None:
                raise ValueError("DA3METRIC needs a focal length: pass focal_px_hint")
            depth = (f * depth / 300.0 * 100.0).astype(np.float32)
        elif name.startswith("DA3NESTED"):
            depth = (depth * 100.0).astype(np.float32)
        return depth, focal

    return run


def depth_scale(
    depth: DepthMap,
    plate_surface: Mask,
    plate_distance_cm: float,
    anchor_xy: tuple[float, float] | None = None,
) -> float:
    """Factor that makes the fitted plate plane lie ``plate_distance_cm`` from the camera.

    The plate plane is fitted to ``plate_surface`` (bare plate pixels) and
    evaluated at ``anchor_xy`` (default: the centroid of ``plate_surface``).
    For an up-to-scale model this is the scale of the frame; for a metric model
    (output already in cm) it should be close to 1, and its distance from 1 is
    a consistency check between the model and the plate-derived distance.

    Raises:
        ValueError: non-positive distance, too few plate pixels, mismatched
            shapes, or a plate plane at non-positive depth.
    """
    if plate_distance_cm <= 0:
        raise ValueError("plate distance must be positive")
    a, b, c = fit_plane(depth, plate_surface)
    if anchor_xy is None:
        ys, xs = np.nonzero(plate_surface)
        anchor_xy = (float(xs.mean()), float(ys.mean()))
    plane_depth = a * anchor_xy[0] + b * anchor_xy[1] + c
    if plane_depth <= 0:
        raise ValueError("plate plane has non-positive depth; the depth map is not usable")
    return plate_distance_cm / plane_depth


def to_centimetres(
    depth: DepthMap,
    plate_surface: Mask,
    plate_distance_cm: float,
    anchor_xy: tuple[float, float] | None = None,
) -> DepthMap:
    """Rescale up-to-scale depth so the plate plane is ``plate_distance_cm`` away.

    Replaces the earlier design that needed a 2 cm calibration block in every
    frame. ``plate_distance_cm`` comes from the plate's known diameter and the
    camera's focal length (``depth.volume.plate_distance_cm``). This assumes
    the model's depth is proportional to true distance (one scale factor, no
    offset); ``eval/`` must check that with a measuring-cup test.

    Raises:
        ValueError: see ``depth_scale``.
    """
    scale = depth_scale(depth, plate_surface, plate_distance_cm, anchor_xy)
    return (depth.astype(np.float64) * scale).astype(np.float32)
