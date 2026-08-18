from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pet_mlops.models.backbone import FEATURE_CHANNELS, FEATURE_CONTRACT


class ClassificationHead(nn.Module):
    """Convert the deepest backbone feature map into class scores.

    Args:
        num_classes: Number of breed scores produced for each image.
        dropout: Probability of dropping pooled features during training.
    """

    requires_feature_contract = FEATURE_CONTRACT

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        """Initialize pooling, regularization, and the output projection.

        Args:
            num_classes: Number of breed logits to produce.
            dropout: Feature dropout probability used during training.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, num_classes)
        )

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        """Produce one vector of unnormalized class scores per image.

        Args:
            features: Backbone feature mapping satisfying ``requires_feature_contract``.

        Returns:
            Classification logits shaped ``[batch, num_classes]``.

        Raises:
            KeyError: If the required ``layer4`` feature is absent.
        """
        return self.classifier(self.pool(features["layer4"]))


class _DecoderBlock(nn.Module):
    """Upsample decoder features and fuse them with one encoder skip connection."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        # TODO what is happening here?
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((value, skip), dim=1))


class SegmentationHead(nn.Module):
    """Compact U-Net-style decoder over the ResNet feature pyramid.
    #TODO what is an encoder and a decoder

    Args:
        num_classes: Number of per-pixel class scores to produce. This project uses
            two: background and foreground.
    """

    requires_feature_contract = FEATURE_CONTRACT

    def __init__(self, num_classes: int) -> None:
        """Initialize the skip-connected decoder and output projection.

        Args:
            num_classes: Number of per-pixel logits to produce.
        """
        super().__init__()
        self.decode3 = _DecoderBlock(FEATURE_CHANNELS["layer4"], FEATURE_CHANNELS["layer3"], 256)
        self.decode2 = _DecoderBlock(256, FEATURE_CHANNELS["layer2"], 128)
        self.decode1 = _DecoderBlock(128, FEATURE_CHANNELS["layer1"], 64)
        self.decode0 = _DecoderBlock(64, FEATURE_CHANNELS["stem"], 32)
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features: Mapping[str, Tensor], output_size: tuple[int, int]) -> Tensor:
        """Decode multi-resolution features into full-size segmentation scores.

        Args:
            features: Backbone feature mapping satisfying ``requires_feature_contract``.
            output_size: Desired output height and width, normally the input image size.

        Returns:
            Segmentation logits shaped ``[batch, num_classes, height, width]``.

        Raises:
            KeyError: If a required pyramid feature is absent.
        """
        value = self.decode3(features["layer4"], features["layer3"])
        value = self.decode2(value, features["layer2"])
        value = self.decode1(value, features["layer1"])
        value = self.decode0(value, features["stem"])
        value = self.output(value)
        return F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
