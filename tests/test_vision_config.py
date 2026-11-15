from __future__ import annotations

from pathlib import Path

import pytest

from vision.train import load_train_config


def test_overrides_win_over_the_committed_config() -> None:
    config = load_train_config(data="/tmp/ds/data.yaml", epochs=3)
    assert config["data"] == "/tmp/ds/data.yaml" and config["epochs"] == 3
    assert config["model"] == "yolo26n-seg.pt" and config["seed"] == 0


def test_config_without_a_model_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("epochs: 1\n")
    with pytest.raises(ValueError):
        load_train_config(bad)
