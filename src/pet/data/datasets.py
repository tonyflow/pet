from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

from pet.data.transforms import PairedTransform

Task = Literal["classification", "segmentation", "multitask"]


class ManifestPetDataset(Dataset):
    """Manifest-ordered Oxford-IIIT Pet dataset for one or both project tasks."""

    def __init__(
        self, root: Path, manifest_path: Path, transform: PairedTransform, task: Task
    ) -> None:
        """Initialize a manifest-filtered view of an official dataset split.

        Args:
            root: Root containing the downloaded Torchvision dataset.
            manifest_path: Parquet manifest selecting and ordering samples.
            transform: Paired image/mask transformation applied per sample.
            task: Target representation to return for each sample.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the manifest or downloaded dataset files are missing.
            KeyError: If a manifest sample ID is absent from the official dataset.
            ValueError: If the manifest spans multiple or unsupported official source splits.
        """
        self.manifest = pl.read_parquet(manifest_path)
        source_split = self.manifest["source_split"].unique().to_list()
        if len(source_split) != 1 or source_split[0] not in {"trainval", "test"}:
            raise ValueError("A manifest must map to exactly one official source split")
        self.dataset = OxfordIIITPet(
            root=root,
            split=source_split[0],
            target_types=["category", "segmentation"],
            download=False,
        )
        index_by_id = {Path(path).stem: index for index, path in enumerate(self.dataset._images)}
        self.indices = [index_by_id[sample_id] for sample_id in self.manifest["sample_id"]]
        self.transform = transform
        self.task = task

    def __len__(self) -> int:
        """Return the number of samples selected by the manifest.

        Args:
            None.

        Returns:
            Number of samples available through this dataset view.

        Raises:
            RuntimeError: If the dataset index has not been initialized.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor | int, str]:
        """Load and transform one task-specific sample and its stable ID.

        Args:
            index: Zero-based position in the manifest-defined dataset view.

        Returns:
            Image tensor, task target, and stable sample ID. The target is a breed integer,
            mask tensor, or multitask target mapping according to ``task``.

        Raises:
            IndexError: If index is outside the dataset bounds.
            OSError: If the source image or mask cannot be read.
            ValueError: If the configured task is unsupported.
        """
        image, (category, mask) = self.dataset[self.indices[index]]
        image_tensor, mask_tensor = self.transform(image, mask)
        sample_id = self.manifest[index, "sample_id"]
        if self.task == "classification":
            return image_tensor, int(category), sample_id
        if self.task == "segmentation":
            return image_tensor, mask_tensor, sample_id

        # Task is training
        return image_tensor, {"category": int(category), "mask": mask_tensor}, sample_id
