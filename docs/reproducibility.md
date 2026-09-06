# Reproducibility controls

- Python is constrained to 3.12 and dependency ranges are recorded in `pyproject.toml`.
- Python, NumPy, and PyTorch RNGs are seeded from configuration.
- Deterministic PyTorch algorithms are enabled by default; cuDNN benchmarking is disabled.
- DataLoader workers receive derived seeds and shuffling uses an explicitly seeded generator.
- Split creation is deterministic, breed-based, and recorded in CSV and Parquet manifests.
- Validation and test transforms contain no random operations.
- Training geometric parameters are sampled once and applied to both image and mask. Masks use
  nearest-neighbor interpolation and are converted from trimap labels to binary foreground masks.
- A model manifest records independently versioned backbone and heads. It references an explicit
  feature contract by ID, file, and SHA-256 digest. The resolved contract is embedded in new
  checkpoints and is validated against backbone tensors at runtime.

Exact floating-point equality across different hardware, PyTorch versions, or kernels is not
guaranteed. Production evaluation must retain environment, Git revision, config, model/data
versions, metrics, per-sample predictions, plots, latency, and GPU-memory measurements; that
evaluation artifact pipeline belongs to a later phase.
