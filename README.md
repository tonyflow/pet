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

`src/pet/models/contracts/` is the typed compatibility package for the shared ResNet-34
feature pyramid and the independently versioned classification and segmentation heads. Training
configuration selects the public registry using the stable `resnet34-v1` ID. Concrete definitions
are grouped by model version, such as `contracts/v1/resnet34.py`. Each contract receives a SHA-256
fingerprint of its canonical dictionary representation. A model or checkpoint is rejected when
its backbone version, resolved contract, selected head version, or class count differs from the
expected manifest. Changing the other task's head version does not invalidate a checkpoint.

Backbone tensors are checked against the resolved contract on the first forward pass. Construct
`PetModel` with `contract_validation="every_forward"` to repeat the checks on every batch during
development or testing.

The classification and segmentation heads have separate typed contracts in the same registry.
They define which backbone features each head consumes and the exact logit shape, layout, dtype,
device, spatial policy, and semantics that it produces. Checkpoints and run artifacts embed
canonical data snapshots of the Python contracts for portability.

The model supports `frozen_backbone` and `fine_tune` training modes. In frozen mode the encoder
parameters and batch-normalization statistics stay fixed; in fine-tuning mode the complete model
is trainable. The epoch trainer uses CUDA float16 autocast and gradient scaling only when both AMP
is requested and the selected device is CUDA, and safely falls back to full precision on CPU.

Small synthetic CPU training tests cover both heads without downloading data:

```bash
.venv/bin/pytest tests/test_models.py tests/test_training.py
```

Run one complete task-specific training, validation, and test job with:

```bash
.venv/bin/pet-train \
  --training-config configs/training/classification_smoke.yaml \
  --data-config configs/data/oxford_pet.yaml \
  --output-root artifacts/training \
  --device cpu
```

Each immutable run directory retains copied configs, provenance and dependency versions,
per-epoch validation predictions, test predictions, best/latest resumable checkpoints, metrics,
latency, peak GPU memory, and an SVG loss curve. Use `--resume <latest.pt>` to continue a run in
a new versioned directory or reopen the same directory. Reopening a run whose configured epochs
are already complete reruns final evaluation and safely regenerates metrics and provenance. The
Runpod configurations use persistent paths under `/workspace`.

## Reproducible containers

Phase 3 adds a CUDA trainer image and a CPU inference image with exact runtime pins. The quickest
portable check builds the inference image and runs both model heads on CPU:

```bash
docker compose build inference-smoke
docker compose run --rm inference-smoke
```

See [docs/containers.md](docs/containers.md) for the NVIDIA workflow, tagging convention, and
the reusable script for credential-safe GitHub Container Registry publishing and downloads.
