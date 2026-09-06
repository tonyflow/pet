from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from torch import Tensor, nn
from torchvision.models import ResNet34_Weights, resnet34

FEATURE_CONTRACT = "resnet34-pyramid-v1"
FEATURE_CHANNELS: Mapping[str, int] = {
    "stem": 64,
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}


class BackboneFeatures(TypedDict):
    """Statically typed keys returned by the version-1 backbone interface."""

    stem: Tensor
    layer1: Tensor
    layer2: Tensor
    layer3: Tensor
    layer4: Tensor


class ResNet34Backbone(nn.Module):
    """ResNet-34 encoder exposing a stable multi-scale feature contract.

    Args:
        pretrained: Whether to initialize the encoder with Torchvision's default
            ImageNet weights. Enabling this may download weights if they are not cached.
    """

    feature_contract = FEATURE_CONTRACT
    feature_channels = FEATURE_CHANNELS

    def __init__(self, pretrained: bool = False) -> None:
        """Initialize the ResNet stages used by the shared encoder.

        Args:
            pretrained: Whether to use Torchvision's default ImageNet weights.
        """
        super().__init__()
        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, image: Tensor) -> BackboneFeatures:
        """Extract spatial features at each ResNet resolution level.

        Args:
            image: Normalized image batch shaped ``[batch, 3, height, width]``.

        Returns:
            Mapping containing ``stem`` and ``layer1`` through ``layer4`` feature tensors.
            Later layers have lower spatial resolution and richer semantic information.
        """
        stem = self.stem(image)
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "stem": stem,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }
