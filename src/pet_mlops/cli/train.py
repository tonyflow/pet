from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from pet_mlops.config import load_data_config
from pet_mlops.reproducibility import seed_everything
from pet_mlops.training.config import load_training_config
from pet_mlops.training.runner import run_training


def resolve_device(value: str) -> torch.device:
    """Resolve a requested device without silently falling back from CUDA."""
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def main() -> None:
    """Run one reproducible classification or segmentation training job."""
    parser = argparse.ArgumentParser(description="Train and evaluate one Oxford Pet task")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    training = load_training_config(args.training_config)
    data = load_data_config(args.data_config)
    seed_everything(data.reproducibility.seed, data.reproducibility.deterministic_algorithms)
    run_name = args.run_name or f"{training.task}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    metrics = run_training(
        training,
        data,
        args.output_root / run_name,
        resolve_device(args.device),
        training_config_path=args.training_config,
        data_config_path=args.data_config,
        resume=args.resume,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
