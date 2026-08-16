from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

from pet_mlops.config import load_data_config
from pet_mlops.data.integrity import validate_manifest
from pet_mlops.data.splits import build_manifests, manifest_sha256


def main() -> None:
    """Download Oxford-IIIT Pet and create deterministic split manifests.

    Args:
        None. Command-line arguments are read from the process argument vector.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid.
        RuntimeError: If the dataset download fails integrity verification.
        OSError: If dataset or manifest files cannot be read or written.
        ValueError: If configuration or annotation data is invalid.
    """
    parser = argparse.ArgumentParser(
        description="Download Oxford-IIIT Pet and build split manifests"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_data_config(args.config)
    if config.dataset.download:
        OxfordIIITPet(root=config.dataset.root, split="trainval", download=True)
        OxfordIIITPet(root=config.dataset.root, split="test", download=True)
    annotation_dir = config.dataset.root / "oxford-iiit-pet" / "annotations"
    manifest = build_manifests(
        annotation_dir,
        config.dataset.manifest_dir,
        config.dataset.validation_fraction,
        config.dataset.split_seed,
    )
    validate_manifest(manifest)
    for split in ("train", "validation", "test"):
        path = config.dataset.manifest_dir / f"{split}.csv"
        count = manifest.filter(manifest["split"] == split).height
        print(f"{split}: {count} sha256={manifest_sha256(path)}")


if __name__ == "__main__":
    main()
