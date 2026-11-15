"""Export a trained checkpoint to ONNX for the rig."""

from __future__ import annotations

from pathlib import Path


def export_onnx(checkpoint: Path, imgsz: int | None = None) -> Path:
    """Export ``checkpoint`` to ONNX and return the path of the ``.onnx`` file.

    ``imgsz`` None keeps the size the model was trained at (a 1024 px model must not be
    exported at 640). ``vision.infer.Segmenter`` reads the input size from the ONNX file.
    """
    from ultralytics import YOLO  # heavy; imported only when exporting

    kwargs = {"imgsz": imgsz} if imgsz is not None else {}
    return Path(YOLO(str(checkpoint)).export(format="onnx", simplify=True, **kwargs))
