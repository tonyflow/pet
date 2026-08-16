# Data layout

- `raw/` contains the downloaded Oxford-IIIT Pet archive contents and is ignored by Git.
- `processed/splits/v1/` contains the deterministic, version-controlled split manifests. CSV is
  the canonical byte-stable representation; Parquet is retained for Polars/PyArrow consumers.

The images and annotations retain their upstream Oxford-IIIT Pet terms. Do not commit raw data.
