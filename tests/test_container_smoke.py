from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pet.cli.container_smoke import resolve_device, run_smoke


def test_cpu_container_smoke_runs_both_heads() -> None:
    result = run_smoke(Path("configs/model/resnet34_v1.yaml"), "cpu", 32)

    assert result["status"] == "ok"
    assert result["device"] == "cpu"
    assert result["classification_shape"] == [1, 37]
    assert result["segmentation_shape"] == [1, 2, 32, 32]


def test_cuda_request_does_not_silently_fall_back(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        resolve_device("cuda")


def test_too_small_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        run_smoke(Path("configs/model/resnet34_v1.yaml"), "cpu", 16)

