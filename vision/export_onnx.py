"""Export a trained checkpoint to ONNX for the rig."""

from __future__ import annotations

from pathlib import Path


def export_onnx(checkpoint: Path, imgsz: int = 640) -> Path:
    """Export ``checkpoint`` to ONNX and return the path of the ``.onnx`` file."""
    from ultralytics import YOLO  # heavy; imported only when exporting

    return Path(YOLO(str(checkpoint)).export(format="onnx", imgsz=imgsz, simplify=True))
