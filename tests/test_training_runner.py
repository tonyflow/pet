from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from pet.config import (
    DataConfig,
    DatasetConfig,
    LoaderConfig,
    ReproducibilityConfig,
    TransformConfig,
)
from pet.training.config import TrainingConfig
from pet.training.runner import run_training


class TinyPetModel(nn.Module):
    def __init__(self, _manifest: object, pretrained_backbone: bool = False) -> None:
        super().__init__()
        del pretrained_backbone
        self.backbone = nn.Conv2d(3, 4, kernel_size=1)
        self.classification = nn.Linear(4, 37)
        self.segmentation = nn.Conv2d(4, 2, kernel_size=1)

    def set_train_mode(self, mode: str) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "fine_tune"

    def forward(self, images: torch.Tensor, task: str) -> torch.Tensor:
        features = self.backbone(images)
        if task == "classification":
            return self.classification(features.mean(dim=(2, 3)))
        return self.segmentation(features)


def _data_config(tmp_path: Path) -> DataConfig:
    manifests = tmp_path / "manifests"
    manifests.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        (manifests / f"{split}.parquet").write_bytes(split.encode())
    return DataConfig(
        dataset=DatasetConfig(
            root=tmp_path / "data",
            manifest_dir=manifests,
            download=False,
            split_version="test-v1",
            validation_fraction=0.2,
            split_seed=7,
        ),
        transforms=TransformConfig(
            image_size=(4, 4),
            horizontal_flip_probability=0,
            rotation_degrees=0,
            scale_range=(1, 1),
            mean=(0, 0, 0),
            std=(1, 1, 1),
        ),
        loader=LoaderConfig(batch_size=2, num_workers=0, pin_memory=False),
        reproducibility=ReproducibilityConfig(seed=7, deterministic_algorithms=True),
    )


def test_runner_writes_versioned_artifact_bundle(tmp_path: Path, monkeypatch) -> None:
    images = torch.randn(2, 3, 4, 4)
    targets = torch.tensor([1, 2])
    batch = (images, targets, ["one", "two"])
    monkeypatch.setattr("pet.training.runner.PetModel", TinyPetModel)
    monkeypatch.setattr(
        "pet.training.runner.build_loaders",
        lambda _config, _task: {"train": [batch], "validation": [batch], "test": [batch]},
    )
    training_path = tmp_path / "training.yaml"
    data_path = tmp_path / "data.yaml"
    training_path.write_text("training\n", encoding="utf-8")
    data_path.write_text("data\n", encoding="utf-8")
    training = TrainingConfig(
        schema_version=1,
        task="classification",
        mode="fine_tune",
        epochs=1,
        learning_rate=0.001,
        weight_decay=0,
        amp=False,
        pretrained_backbone=False,
        model_manifest=Path("configs/model/resnet34_v1.yaml"),
    )
    run_dir = tmp_path / "run"

    metrics = run_training(
        training,
        _data_config(tmp_path),
        run_dir,
        torch.device("cpu"),
        training_config_path=training_path,
        data_config_path=data_path,
    )

    assert metrics["peak_gpu_memory_bytes"] == 0
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "latest.pt").is_file()
    assert (run_dir / "predictions" / "test.jsonl").is_file()
    assert (run_dir / "learning-curves.svg").is_file()
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["task"] == "classification"
    assert provenance["data_split_version"] == "test-v1"


def test_runner_resumes_after_last_completed_epoch(tmp_path: Path, monkeypatch) -> None:
    images = torch.randn(2, 3, 4, 4)
    targets = torch.tensor([1, 2])
    batch = (images, targets, ["one", "two"])
    monkeypatch.setattr("pet.training.runner.PetModel", TinyPetModel)
    monkeypatch.setattr(
        "pet.training.runner.build_loaders",
        lambda _config, _task: {"train": [batch], "validation": [batch], "test": [batch]},
    )
    training_path = tmp_path / "training.yaml"
    data_path = tmp_path / "data.yaml"
    training_path.write_text("training\n", encoding="utf-8")
    data_path.write_text("data\n", encoding="utf-8")
    base = TrainingConfig(
        schema_version=1,
        task="classification",
        mode="fine_tune",
        epochs=1,
        learning_rate=0.001,
        weight_decay=0,
        amp=False,
        pretrained_backbone=False,
        model_manifest=Path("configs/model/resnet34_v1.yaml"),
    )
    first_run = tmp_path / "first"
    run_training(
        base,
        _data_config(tmp_path),
        first_run,
        torch.device("cpu"),
        training_config_path=training_path,
        data_config_path=data_path,
    )
    resumed = TrainingConfig(**{**base.__dict__, "epochs": 2})
    second_run = tmp_path / "second"

    run_training(
        resumed,
        _data_config(tmp_path),
        second_run,
        torch.device("cpu"),
        training_config_path=training_path,
        data_config_path=data_path,
        resume=first_run / "checkpoints" / "latest.pt",
    )

    history = json.loads((second_run / "history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [2]
