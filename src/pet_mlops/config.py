from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset locations and deterministic split settings."""

    root: Path
    manifest_dir: Path
    download: bool
    split_version: str
    validation_fraction: float
    split_seed: int


@dataclass(frozen=True)
class TransformConfig:
    """Image normalization and paired augmentation settings."""

    image_size: tuple[int, int]
    horizontal_flip_probability: float
    rotation_degrees: float
    scale_range: tuple[float, float]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class LoaderConfig:
    """PyTorch data-loader batching and worker settings."""

    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Random seed and deterministic-algorithm settings."""

    seed: int
    deterministic_algorithms: bool


@dataclass(frozen=True)
class DataConfig:
    """Complete validated data-pipeline configuration."""

    dataset: DatasetConfig
    transforms: TransformConfig
    loader: LoaderConfig
    reproducibility: ReproducibilityConfig


def _tuple(values: list[Any], length: int, name: str) -> tuple[Any, ...]:
    """Convert a list to a tuple after validating its required length."""
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(values)


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate a version-one data configuration from YAML.

    Args:
        path: Location of the YAML configuration file.

    Returns:
        A typed, immutable data configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If a required configuration key is absent.
        TypeError: If a configuration section has an incompatible shape.
        ValueError: If the schema version or a constrained value is invalid.
        yaml.YAMLError: If the file does not contain valid YAML.
    """
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported config schema_version")
    dataset = raw["dataset"]
    transforms = raw["transforms"]
    loader = raw["loader"]
    repro = raw["reproducibility"]
    validation_fraction = float(dataset["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    scale_range = _tuple(transforms["scale_range"], 2, "scale_range")
    if not 0 < scale_range[0] <= scale_range[1]:
        raise ValueError("scale_range must be positive and ordered")
    return DataConfig(
        dataset=DatasetConfig(
            root=Path(dataset["root"]),
            manifest_dir=Path(dataset["manifest_dir"]),
            download=bool(dataset["download"]),
            split_version=str(dataset["split_version"]),
            validation_fraction=validation_fraction,
            split_seed=int(dataset["split_seed"]),
        ),
        transforms=TransformConfig(
            image_size=_tuple(transforms["image_size"], 2, "image_size"),
            horizontal_flip_probability=float(transforms["horizontal_flip_probability"]),
            rotation_degrees=float(transforms["rotation_degrees"]),
            scale_range=scale_range,
            mean=_tuple(transforms["mean"], 3, "mean"),
            std=_tuple(transforms["std"], 3, "std"),
        ),
        loader=LoaderConfig(**loader),
        reproducibility=ReproducibilityConfig(**repro),
    )
