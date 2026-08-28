from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from pet.config import DataConfig
from pet.data.loaders import build_loaders
from pet.models import PetModel, load_model_manifest
from pet.training.checkpoints import load_checkpoint, save_checkpoint
from pet.training.config import TrainingConfig
from pet.training.engine import train_one_epoch, trainable_parameters
from pet.training.evaluation import EvaluationResult, evaluate


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    embedded_revision = os.environ.get("PET_RUN_GIT_REVISION") or os.environ.get(
        "PET_GIT_REVISION"
    )
    if embedded_revision:
        return embedded_revision
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Invalid training history: {path}")
    return value


def _initialize_run_directory(
    run_dir: Path,
    training: TrainingConfig,
    *,
    training_config_path: Path,
    data_config_path: Path,
    resume: Path | None,
) -> list[dict[str, Any]]:
    if run_dir.exists():
        if resume is None:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Pass --resume with its latest "
                "checkpoint to continue or finalize it."
            )
        required = [
            run_dir / "configs",
            run_dir / "checkpoints",
            run_dir / "predictions",
        ]
        missing = [str(path) for path in required if not path.is_dir()]
        if missing:
            raise ValueError(f"Existing run directory is incomplete: {', '.join(missing)}")
        return _load_history(run_dir / "history.json")

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "configs").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "predictions").mkdir()
    shutil.copy2(training_config_path, run_dir / "configs" / training_config_path.name)
    shutil.copy2(data_config_path, run_dir / "configs" / data_config_path.name)
    shutil.copy2(training.model_manifest, run_dir / "configs" / training.model_manifest.name)
    return []


def _curve_svg(history: list[dict[str, Any]]) -> str:
    width, height, pad = 640, 360, 45
    losses = [float(row[key]) for row in history for key in ("train_loss", "validation_loss")]
    high, low = max(losses), min(losses)
    span = max(high - low, 1e-12)

    def points(key: str) -> str:
        count = max(len(history) - 1, 1)
        return " ".join(
            f"{pad + index * (width - 2 * pad) / count:.1f},"
            f"{height - pad - (float(row[key]) - low) * (height - 2 * pad) / span:.1f}"
            for index, row in enumerate(history)
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{points("train_loss")}"/>'
        f'<polyline fill="none" stroke="#dc2626" stroke-width="3" '
        f'points="{points("validation_loss")}"/>'
        f'<text x="{pad}" y="25" font-family="sans-serif">'
        "training (blue) / validation (red) loss</text>"
        f'<text x="{pad}" y="{height - 10}" font-family="sans-serif">epoch</text>'
        "</svg>\n"
    )


def run_training(
    training: TrainingConfig,
    data: DataConfig,
    run_dir: Path,
    device: torch.device,
    *,
    training_config_path: Path,
    data_config_path: Path,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Train, validate, checkpoint, and test one independently versioned task head."""
    history = _initialize_run_directory(
        run_dir,
        training,
        training_config_path=training_config_path,
        data_config_path=data_config_path,
        resume=resume,
    )

    manifest = load_model_manifest(training.model_manifest)
    model = PetModel(manifest, pretrained_backbone=training.pretrained_backbone).to(device)
    model.set_train_mode(training.mode)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    start_epoch = 1
    if resume is not None:
        start_epoch = (
            load_checkpoint(
                resume, model, manifest, training.task, optimizer=optimizer, map_location=device
            )
            + 1
        )
    loaders = build_loaders(data, training.task)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_loss = min(
        (float(row["validation_loss"]) for row in history), default=float("inf")
    )
    best_path = run_dir / "checkpoints" / "best.pt"
    latest_path = run_dir / "checkpoints" / "latest.pt"
    for epoch in range(start_epoch, training.epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, training.task, device, amp=training.amp
        )
        validation = evaluate(
            model, loaders["validation"], training.task, device, amp=training.amp
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"validation_{key}": value for key, value in validation.metrics.items()},
        }
        history.append(row)
        _write_json(run_dir / "history.json", history)
        _write_jsonl(
            run_dir / "predictions" / f"validation-epoch-{epoch:03d}.jsonl",
            validation.predictions,
        )
        save_checkpoint(
            latest_path, model, manifest, training.task, epoch=epoch, optimizer=optimizer
        )
        if float(validation.metrics["loss"]) < best_loss:
            best_loss = float(validation.metrics["loss"])
            save_checkpoint(
                best_path, model, manifest, training.task, epoch=epoch, optimizer=optimizer
            )

    if not history:
        raise ValueError("No completed epochs are available for final evaluation")
    load_checkpoint(best_path, model, manifest, training.task, map_location=device)
    test_result: EvaluationResult = evaluate(
        model, loaders["test"], training.task, device, amp=training.amp
    )
    _write_jsonl(run_dir / "predictions" / "test.jsonl", test_result.predictions)
    (run_dir / "learning-curves.svg").write_text(_curve_svg(history), encoding="utf-8")

    manifest_hashes = {
        split: _sha256(data.dataset.manifest_dir / f"{split}.parquet")
        for split in ("train", "validation", "test")
    }
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "container_build_git_revision": os.environ.get("PET_GIT_REVISION"),
        "container_image": os.environ.get("PET_CONTAINER_IMAGE"),
        "container_image_digest": os.environ.get("PET_CONTAINER_IMAGE_DIGEST"),
        "task": training.task,
        "model_manifest": manifest.to_dict(),
        "data_split_version": data.dataset.split_version,
        "data_manifest_sha256": manifest_hashes,
        "device": str(device),
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("pet", "torch", "torchvision")},
    }
    metrics = {
        "best_validation_loss": best_loss,
        "test": test_result.metrics,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    _write_json(run_dir / "provenance.json", provenance)
    _write_json(run_dir / "metrics.json", metrics)
    return metrics
