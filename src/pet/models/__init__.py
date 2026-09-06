"""Versioned model components and compatibility contracts."""

from pet.models.backbone import BackboneFeatures, ResNet34Backbone
from pet.models.heads import ClassificationHead, SegmentationHead
from pet.models.manifest import (
    FeatureContract,
    FeatureTensorSpec,
    HeadContract,
    ModelManifest,
    load_model_manifest,
)
from pet.models.multitask import ContractValidationMode, PetModel

__all__ = [
    "BackboneFeatures",
    "ClassificationHead",
    "ContractValidationMode",
    "FeatureContract",
    "FeatureTensorSpec",
    "HeadContract",
    "ModelManifest",
    "PetModel",
    "ResNet34Backbone",
    "SegmentationHead",
    "load_model_manifest",
]
