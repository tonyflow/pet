import random

import numpy as np
import torch
from PIL import Image

from pet_mlops.config import TransformConfig
from pet_mlops.data.transforms import PairedTransform


def _config() -> TransformConfig:
    """Return a small transform configuration suitable for synthetic images."""
    return TransformConfig(
        image_size=(32, 32),
        horizontal_flip_probability=0.5,
        rotation_degrees=10,
        scale_range=(0.9, 1.1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )


def _pair() -> tuple[Image.Image, Image.Image]:
    """Create an aligned synthetic image and trimap mask pair."""
    mask = np.full((20, 20), 2, dtype=np.uint8)
    mask[5:15, 5:15] = 1
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = 255
    return Image.fromarray(image), Image.fromarray(mask)


def test_evaluation_transform_is_deterministic() -> None:
    """Keep evaluation outputs unchanged when the global RNG seed changes."""
    transform = PairedTransform(_config(), training=False)
    image, mask = _pair()
    first = transform(image, mask)
    for seed in range(5):
        random.seed(seed)
        second = transform(image, mask)
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])


def test_training_geometry_stays_synchronized() -> None:
    """Apply identical sampled geometry to the training image and mask."""
    transform = PairedTransform(_config(), training=True)
    image, mask = _pair()
    random.seed(8)
    image_tensor, mask_tensor = transform(image, mask)
    bright = image_tensor.mean(dim=0) > 0.5
    intersection = (bright & mask_tensor.bool()).sum()
    union = (bright | mask_tensor.bool()).sum()
    # Bilinear image interpolation softens the shared boundary while the label mask stays discrete.
    assert intersection / union > 0.9
    assert set(torch.unique(mask_tensor).tolist()).issubset({0, 1})
