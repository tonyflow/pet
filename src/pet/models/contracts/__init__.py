"""Public registry for versioned typed model contracts."""

from pet.models.contracts.v1.resnet34 import (
    CLASSIFICATION_HEAD_V1,
    RESNET34_PYRAMID_V1,
    RESNET34_V1,
    SEGMENTATION_HEAD_V1,
)
from pet.models.manifest import ModelManifest

MODEL_MANIFESTS: dict[str, ModelManifest] = {"resnet34-v1": RESNET34_V1}

__all__ = [
    "CLASSIFICATION_HEAD_V1",
    "MODEL_MANIFESTS",
    "RESNET34_PYRAMID_V1",
    "RESNET34_V1",
    "SEGMENTATION_HEAD_V1",
]
