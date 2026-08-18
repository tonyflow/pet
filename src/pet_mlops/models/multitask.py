from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from pet_mlops.models.backbone import ResNet34Backbone
from pet_mlops.models.heads import ClassificationHead, SegmentationHead
from pet_mlops.models.manifest import ModelManifest

TrainMode = Literal["frozen_backbone", "fine_tune"]


class PetModel(nn.Module):
    """Compose one shared image encoder with two independently versioned heads.

    Args:
        manifest: Compatibility metadata defining backbone and head outputs.
        pretrained_backbone: Whether to initialize ResNet-34 with ImageNet weights.

    Raises:
        ValueError: If the manifest requests an unsupported feature contract.
    """

    def __init__(self, manifest: ModelManifest, pretrained_backbone: bool = False) -> None:
        """Initialize and validate the shared backbone and both heads.

        Args:
            manifest: Compatibility metadata defining the component versions and outputs.
            pretrained_backbone: Whether to start from ImageNet backbone weights.

        Raises:
            ValueError: If the manifest is invalid or requests an unsupported contract.
        """
        super().__init__()
        manifest.validate()
        if manifest.backbone.feature_contract != ResNet34Backbone.feature_contract:
            raise ValueError(
                f"Manifest contract {manifest.backbone.feature_contract!r} is not implemented; "
                f"expected {ResNet34Backbone.feature_contract!r}"
            )
        self.manifest = manifest
        self.backbone = ResNet34Backbone(pretrained=pretrained_backbone)
        self.classification = ClassificationHead(manifest.classification.num_classes)
        self.segmentation = SegmentationHead(manifest.segmentation.num_classes)

    def set_train_mode(self, mode: TrainMode) -> None:
        """Freeze or unfreeze all shared-backbone parameters.

        Args:
            mode: ``frozen_backbone`` to train heads only, or ``fine_tune`` to update
                the backbone as well.

        Raises:
            ValueError: If ``mode`` is unsupported.
        """
        if mode not in {"frozen_backbone", "fine_tune"}:
            raise ValueError(f"Unsupported train mode: {mode}")
        requires_grad = mode == "fine_tune"
        for parameter in self.backbone.parameters():
            #TODO is parameters the number of weights in all layers of the network?
            parameter.requires_grad = requires_grad

    def train(self, mode: bool = True) -> PetModel:
        """Set training state while keeping a frozen backbone deterministic.

        Args:
            mode: Whether the model should enter training mode.

        Returns:
            This model, matching the standard PyTorch ``Module.train`` API.
        """
        super().train(mode)
        if mode and not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, image: Tensor, task: Literal["classification", "segmentation"]) -> Tensor:
        """Run the shared backbone followed by one selected task head.

        Args:
            image: Normalized image batch shaped ``[batch, 3, height, width]``.
            task: Head to execute.

        Returns:
            Classification logits or full-resolution segmentation logits.

        Raises:
            ValueError: If ``task`` is unsupported.
        """
        features = self.backbone(image)
        if task == "classification":
            return self.classification(features)
        if task == "segmentation":
            return self.segmentation(features, image.shape[-2:])
        raise ValueError(f"Unsupported task: {task}")
