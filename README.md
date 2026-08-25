# Classification & Segmentation

Use Oxford-IIIT Pet project with two tasks:
1. 37-class breed classification and 
2. foreground segmentation. 

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

## Versioned models and training

`configs/model/resnet34_v1.yaml` is the compatibility boundary between the shared ResNet-34
feature pyramid and the independently versioned classification and segmentation heads. A model
or checkpoint is rejected before weight loading when its backbone version, feature contract,
selected head version, or class count differs from the expected manifest. Changing the other
task's head version does not invalidate a checkpoint.

The model supports `frozen_backbone` and `fine_tune` training modes. In frozen mode the encoder
parameters and batch-normalization statistics stay fixed; in fine-tuning mode the complete model
is trainable. The epoch trainer uses CUDA float16 autocast and gradient scaling only when both AMP
is requested and the selected device is CUDA, and safely falls back to full precision on CPU.

Small synthetic CPU training tests cover both heads without downloading data:

```bash
.venv/bin/pytest tests/test_models.py tests/test_training.py
```

## Reproducible containers

Phase 3 adds a CUDA trainer image and a CPU inference image with exact runtime pins. The quickest
portable check builds the inference image and runs both model heads on CPU:

```bash
docker compose build inference-smoke
docker compose run --rm inference-smoke
```

See [docs/containers.md](docs/containers.md) for the NVIDIA workflow, tagging convention, and
the reusable script for credential-safe GitHub Container Registry publishing and downloads.
