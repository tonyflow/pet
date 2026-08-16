from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pet_mlops.config import load_data_config
from pet_mlops.data.integrity import save_overlay
from pet_mlops.data.loaders import build_loaders
from pet_mlops.reproducibility import seed_everything


def main() -> None:
    """Run CPU data-loader checks and save one mask overlay per split.

    Args:
        None. Command-line arguments are read from the process argument vector.

    Returns:
        None.

    Raises:
        AssertionError: If a loaded batch violates the expected CPU data contract.
        FileNotFoundError: If configuration, manifests, or dataset files are missing.
        OSError: If a visual-integrity artifact cannot be written.
        SystemExit: If command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Run CPU loader and image/mask integrity checks")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_data_config(args.config)
    seed_everything(config.reproducibility.seed, config.reproducibility.deterministic_algorithms)
    loaders = build_loaders(config, task="segmentation")
    for split, loader in loaders.items():
        images, masks, sample_ids = next(iter(loader))
        assert images.device.type == "cpu"
        assert images.shape[-2:] == masks.shape[-2:] == config.transforms.image_size
        assert set(torch.unique(masks).tolist()).issubset({0, 1})
        # Undo ImageNet normalization for a human-readable integrity artifact.
        mean = torch.tensor(config.transforms.mean).view(3, 1, 1)
        std = torch.tensor(config.transforms.std).view(3, 1, 1)
        save_overlay(images[0] * std + mean, masks[0], args.output / f"{split}-{sample_ids[0]}.png")
        print(f"{split}: images={tuple(images.shape)} masks={tuple(masks.shape)}")


if __name__ == "__main__":
    main()
