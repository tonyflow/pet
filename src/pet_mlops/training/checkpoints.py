from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from pet_mlops.models.manifest import ModelManifest, TaskName, manifests_compatible

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    manifest: ModelManifest,
    task: TaskName,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save task weights with their exact compatibility manifest.

    Args:
        path: Destination checkpoint file.
        model: Model containing ``backbone`` and task-named head modules.
        manifest: Model metadata to embed for compatibility validation.
        task: Head whose weights should be saved.
        epoch: Last completed training epoch.
        optimizer: Optional optimizer whose state should support resuming training.

    Raises:
        ValueError: If the selected model manifest is invalid.
        OSError: If the checkpoint cannot be written.
    """
    manifest.validate(task)
    state: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task": task,
        "epoch": epoch,
        "manifest": manifest.to_dict(),
        "backbone_state_dict": model.backbone.state_dict(),
        "head_state_dict": getattr(model, task).state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, Path(path))


def _manifest_from_checkpoint(raw: dict[str, Any]) -> ModelManifest:
    backbone = raw["backbone"]
    heads = raw["heads"]
    from pet_mlops.models.manifest import BackboneManifest, HeadManifest

    return ModelManifest(
        schema_version=int(raw["schema_version"]),
        backbone=BackboneManifest(**backbone),
        classification=HeadManifest(**heads["classification"]),
        segmentation=HeadManifest(**heads["segmentation"]),
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    expected_manifest: ModelManifest,
    task: TaskName,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Validate metadata before loading backbone and task-head weights.

    Args:
        path: Checkpoint file to load.
        model: Destination model containing ``backbone`` and task-named head modules.
        expected_manifest: Compatibility metadata required by the current run.
        task: Head expected in the checkpoint.
        optimizer: Optional optimizer into which stored state should be restored.
        map_location: Device mapping passed to ``torch.load``.

    Returns:
        Last completed epoch recorded in the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If its schema, task, backbone, or selected head is incompatible.
        KeyError: If required checkpoint data is absent.
        RuntimeError: If stored tensor shapes do not match the model.
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema_version")
    if checkpoint.get("task") != task:
        raise ValueError(f"Checkpoint task {checkpoint.get('task')!r} does not match {task!r}")
    actual_manifest = _manifest_from_checkpoint(checkpoint["manifest"])
    manifests_compatible(expected_manifest, actual_manifest, task)
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
    getattr(model, task).load_state_dict(checkpoint["head_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"])
