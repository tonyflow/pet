from __future__ import annotations

import hashlib
import random
from pathlib import Path

import polars as pl

MANIFEST_COLUMNS = ["sample_id", "breed_id", "species_id", "source_split", "split"]


def parse_official_list(path: Path, source_split: str) -> pl.DataFrame:
    """Parse an official Oxford-IIIT Pet list into a zero-based manifest.

    Args:
        path: Official annotation-list file to parse.
        source_split: Upstream partition name assigned to every parsed sample.

    Returns:
        A Polars table containing sample IDs and zero-based labels.

    Raises:
        FileNotFoundError: If the annotation list does not exist.
        ValueError: If an annotation row is malformed or contains non-integer labels.
    """
    rows: list[dict[str, int | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        sample_id, breed_id, species_id, _ = line.split()
        rows.append(
            {
                "sample_id": sample_id,
                "breed_id": int(breed_id) - 1,
                "species_id": int(species_id) - 1,
                "source_split": source_split,
                "split": source_split,
            }
        )
    return pl.DataFrame(rows, schema=MANIFEST_COLUMNS, orient="row")


def breed_based_train_validation_split(
    trainval: pl.DataFrame, validation_fraction: float, seed: int
) -> pl.DataFrame:
    """Assign trainval samples to deterministic breed-based splits.

    Args:
        trainval: Parsed official trainval manifest.
        validation_fraction: Fraction of each breed assigned to validation.
        seed: Base seed used for deterministic per-breed shuffling.

    Returns:
        The manifest with each sample assigned to train or validation.

    Raises:
        polars.exceptions.ColumnNotFoundError: If required manifest columns are absent.
        ValueError: If assignments cannot be mapped back to all sample IDs.
    """
    assignments: dict[str, str] = {}
    for breed_id in trainval["breed_id"].unique().sort().to_list():
        sample_ids = sorted(trainval.filter(pl.col("breed_id") == breed_id)["sample_id"].to_list())
        random.Random(seed + int(breed_id)).shuffle(sample_ids)
        validation_count = max(1, round(len(sample_ids) * validation_fraction))
        validation_ids = set(sample_ids[:validation_count])
        assignments.update(
            {
                sample_id: "validation" if sample_id in validation_ids else "train"
                for sample_id in sample_ids
            }
        )
    return trainval.with_columns(
        pl.col("sample_id").replace_strict(assignments).alias("split")
    ).sort("sample_id")


def build_manifests(
    annotation_dir: Path, output_dir: Path, validation_fraction: float, seed: int
) -> pl.DataFrame:
    """Build, validate, and persist train, validation, and test manifests.

    Args:
        annotation_dir: Directory containing official trainval and test lists.
        output_dir: Destination for CSV and Parquet manifests.
        validation_fraction: Fraction of each trainval breed used for validation.
        seed: Seed controlling deterministic split assignment.

    Returns:
        A combined manifest containing all generated splits.

    Raises:
        FileNotFoundError: If either official annotation list is missing.
        OSError: If the output directory or manifest files cannot be written.
        ValueError: If official annotations contain duplicate sample IDs or malformed rows.
    """
    trainval = parse_official_list(annotation_dir / "trainval.txt", "trainval")
    test = parse_official_list(annotation_dir / "test.txt", "test").with_columns(
        pl.lit("test").alias("split")
    )
    manifest = pl.concat(
        [breed_based_train_validation_split(trainval, validation_fraction, seed), test]
    ).sort(["split", "sample_id"])
    if manifest["sample_id"].n_unique() != manifest.height:
        raise ValueError("Duplicate sample IDs found in official annotations")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        split_frame = manifest.filter(pl.col("split") == split).select(MANIFEST_COLUMNS)
        split_frame.write_csv(output_dir / f"{split}.csv")
        split_frame.write_parquet(output_dir / f"{split}.parquet")
    return manifest


def manifest_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a manifest file.

    Args:
        path: Manifest file whose bytes will be hashed.

    Returns:
        The lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        OSError: If the manifest file cannot be read.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
