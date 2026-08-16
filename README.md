# Pet MLOps

Phase 1 foundations for a reproducible Oxford-IIIT Pet project with two tasks:
37-class breed classification and foreground segmentation. The project targets Python 3.12,
PyTorch/Torchvision, and Polars (with PyArrow for Parquet interoperability).

## Local setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
```

Prepare the official dataset and immutable split manifests:

```bash
.venv/bin/pet-prepare-data --config configs/data/oxford_pet.yaml
```

Run a CPU-only loader and visual-integrity smoke check:

```bash
.venv/bin/pet-data-smoke --config configs/data/oxford_pet.yaml --output artifacts/data-smoke
```

The official `trainval` partition is split deterministically by breed into training
and validation subsets. The official test partition remains untouched. Re-running preparation with
the same seed produces byte-stable CSV manifests; Parquet copies are also retained for efficient
Polars workflows. Augmentation is applied to training only, and every geometric operation uses the
same sampled parameters for each image/mask pair.

See [docs/reproducibility.md](docs/reproducibility.md) for controls and limitations.
