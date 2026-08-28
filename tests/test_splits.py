from pathlib import Path

import polars as pl

from pet.data.splits import build_manifests, manifest_sha256


def _write_official_lists(root: Path) -> None:
    """Write small synthetic files matching the official annotation format."""
    root.mkdir()
    lines = []
    for breed in range(1, 4):
        for index in range(10):
            lines.append(f"breed{breed}_{index} {breed} {(breed % 2) + 1} 1")
    (root / "trainval.txt").write_text("\n".join(lines), encoding="utf-8")
    (root / "test.txt").write_text("test_0 1 1 1\n", encoding="utf-8")


def test_split_is_breed_based_disjoint_and_deterministic(tmp_path: Path) -> None:
    """Preserve every sample and breed while producing stable split files."""
    annotations = tmp_path / "annotations"
    _write_official_lists(annotations)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_manifests(annotations, first, 0.2, 1729)
    build_manifests(annotations, second, 0.2, 1729)
    assert manifest.height == manifest["sample_id"].n_unique() == 31
    validation = manifest.filter(pl.col("split") == "validation")
    assert validation.group_by("breed_id").len()["len"].to_list() == [2, 2, 2]
    for split in ("train", "validation", "test"):
        assert manifest_sha256(first / f"{split}.csv") == manifest_sha256(second / f"{split}.csv")
