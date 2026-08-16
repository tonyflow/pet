from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from pet_mlops.config import TransformConfig


def trimap_to_foreground(mask: torch.Tensor) -> torch.Tensor:
    """Convert Oxford trimap labels to a binary foreground mask.

    Args:
        mask: Tensor containing Oxford trimap values 1, 2, and 3.

    Returns:
        Integer tensor containing zero for background and one for pet or border.

    Raises:
        RuntimeError: If the input tensor cannot be compared or converted to integer labels.
    """
    # Oxford trimap: 1=pet, 2=background, 3=border. Border is retained as foreground.
    return (mask != 2).to(torch.long)


@dataclass(frozen=True)
class PairedTransform:
    config: TransformConfig
    training: bool

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform an image and mask with shared training geometry.

        Args:
            image: PIL image to resize, augment when training, and normalize.
            mask: PIL trimap receiving the same sampled geometric operations.

        Returns:
            A normalized image tensor and aligned binary mask tensor.

        Raises:
            TypeError: If image or mask is not compatible with PIL/Torchvision operations.
            ValueError: If configured transform parameters are invalid.
        """
        image = image.convert("RGB")
        if self.training:
            if random.random() < self.config.horizontal_flip_probability:
                image, mask = F.hflip(image), F.hflip(mask)
            angle = random.uniform(-self.config.rotation_degrees, self.config.rotation_degrees)
            scale = random.uniform(*self.config.scale_range)
            image = F.affine(
                image,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = F.affine(
                mask,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=2,
            )
        image = F.resize(
            image, self.config.image_size, interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        mask = F.resize(mask, self.config.image_size, interpolation=InterpolationMode.NEAREST)
        image_tensor = F.normalize(F.to_tensor(image), self.config.mean, self.config.std)
        mask_tensor = F.pil_to_tensor(mask).squeeze(0)
        return image_tensor, trimap_to_foreground(mask_tensor)
