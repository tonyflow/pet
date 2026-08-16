from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from PIL import Image, ImageDraw


def validate_manifest(frame: pl.DataFrame) -> None:
    """Validate required columns, sample IDs, and breed-label bounds.

    Args:
        frame: Combined or split manifest to validate.

    Returns:
        None.

    Raises:
        ValueError: If columns are missing, sample IDs are invalid, or labels are out of range.
    """
    required = {"sample_id", "breed_id", "species_id", "source_split", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(frame.columns))}")
    if frame["sample_id"].null_count() or frame["sample_id"].n_unique() != frame.height:
        raise ValueError("sample_id must be non-null and unique")
    if not set(frame["breed_id"].unique().to_list()).issubset(set(range(37))):
        raise ValueError("breed_id values must be zero-based values in [0, 36]")


def save_overlay(image: torch.Tensor, mask: torch.Tensor, path: Path) -> None:
    """Save a translucent foreground-mask overlay for visual inspection.

    Args:
        image: Unnormalized three-channel image tensor with values expected in [0, 1].
        mask: Two-dimensional binary mask aligned with the image.
        path: Destination image path, including its filename.

    Returns:
        None.

    Raises:
        OSError: If the destination directory or image file cannot be written.
        ValueError: If image or mask shapes cannot be converted into an aligned overlay.
    """
    normalized = image.detach().cpu().clamp(0, 1)
    canvas = Image.fromarray((normalized.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mask_values = mask.detach().cpu().flatten().tolist()
    width, _ = canvas.size
    for index, value in enumerate(mask_values):
        if value:
            draw.point((index % width, index // width), fill=(255, 0, 128, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB").save(path)
