from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pet.models.manifest import TaskName
from pet.models.multitask import TrainMode


@dataclass(frozen=True)
class TrainingConfig:
    """Validated settings for one task-specific training run."""

    schema_version: int
    task: TaskName
    mode: TrainMode
    epochs: int
    learning_rate: float
    weight_decay: float
    amp: bool
    pretrained_backbone: bool
    model_manifest: Path

    def validate(self) -> None:
        """Validate supported choices and numeric constraints.

        Raises:
            ValueError: If the schema, task, mode, or optimization values are invalid.
        """
        if self.schema_version != 1:
            raise ValueError(f"Unsupported training schema_version: {self.schema_version}")
        if self.task not in {"classification", "segmentation"}:
            raise ValueError(f"Unsupported task: {self.task}")
        if self.mode not in {"frozen_backbone", "fine_tune"}:
            raise ValueError(f"Unsupported training mode: {self.mode}")
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "epochs and learning_rate must be positive; weight_decay cannot be negative"
            )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load and validate a task training configuration from YAML.

    Args:
        path: YAML training configuration location.

    Returns:
        Validated immutable training configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML root is not a mapping.
        KeyError: If a required configuration value is absent.
        ValueError: If a configuration value is invalid.
        yaml.YAMLError: If the file is not valid YAML.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Training config must be a YAML mapping")
    config = TrainingConfig(
        schema_version=int(raw["schema_version"]),
        task=raw["task"],
        mode=raw["mode"],
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        amp=bool(raw["amp"]),
        pretrained_backbone=bool(raw["pretrained_backbone"]),
        model_manifest=Path(raw["model_manifest"]),
    )
    config.validate()
    return config
