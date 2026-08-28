from __future__ import annotations

from torch.utils.data import DataLoader

from pet.config import DataConfig
from pet.data.datasets import ManifestPetDataset, Task
from pet.data.transforms import PairedTransform
from pet.reproducibility import seed_worker, seeded_generator


def build_loaders(config: DataConfig, task: Task) -> dict[str, DataLoader]:
    """Build reproducible PyTorch loaders for all three dataset splits.

    Args:
        config: Dataset, transform, loader, and reproducibility settings.
        task: Target representation requested from each dataset.

    Returns:
        Mapping from train, validation, and test names to DataLoaders.

    Raises:
        FileNotFoundError: If a split manifest or downloaded dataset file is missing.
        ValueError: If a manifest source split or task is unsupported.
    """
    loaders = {}
    for split in ("train", "validation", "test"):
        dataset = ManifestPetDataset(
            root=config.dataset.root,
            manifest_path=config.dataset.manifest_dir / f"{split}.parquet",
            transform=PairedTransform(config.transforms, training=split == "train"),
            task=task,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.loader.batch_size,
            shuffle=split == "train",
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            worker_init_fn=seed_worker,
            generator=seeded_generator(config.reproducibility.seed),
        )
    return loaders
