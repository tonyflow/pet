from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from pet.models import PetModel, load_model_manifest
from pet.training.config import load_training_config
from pet.training.engine import train_one_epoch, trainable_parameters


@pytest.mark.parametrize(
    ("task", "target"),
    [
        ("classification", torch.tensor([1, 2])),
        ("segmentation", torch.randint(0, 2, (2, 32, 32))),
    ],
)
def test_cpu_smoke_training_for_each_head(task, target) -> None:
    torch.manual_seed(7)
    manifest = load_model_manifest(Path("configs/model/resnet34_v1.yaml"))
    model = PetModel(manifest)
    model.set_train_mode("frozen_backbone")
    images = torch.randn(2, 3, 32, 32)
    samples = [(images[index], target[index], str(index)) for index in range(2)]
    loader = DataLoader(samples, batch_size=2)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=1e-3)

    loss = train_one_epoch(
        model, loader, optimizer, task, torch.device("cpu"), amp=True
    )
    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_training_config() -> None:
    config = load_training_config(Path("configs/training/classification_smoke.yaml"))
    assert config.task == "classification"
    assert config.mode == "frozen_backbone"
    assert config.amp is False
