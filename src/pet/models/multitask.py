from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from pet.models.backbone import ResNet34Backbone
from pet.models.heads import ClassificationHead, SegmentationHead
from pet.models.manifest import ModelManifest

TrainMode = Literal["frozen_backbone", "fine_tune"]
ContractValidationMode = Literal["first_forward", "every_forward"]


class PetModel(nn.Module):
    """Compose one shared image encoder with two independently versioned heads.

    Args:
        manifest: Compatibility metadata defining backbone and head outputs.
        pretrained_backbone: Whether to initialize ResNet-34 with ImageNet weights.

    Raises:
        ValueError: If the manifest requests an unsupported feature contract.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        pretrained_backbone: bool = False,
        contract_validation: ContractValidationMode = "first_forward",
    ) -> None:
        """Initialize and validate the shared backbone and both heads.

        Args:
            manifest: Compatibility metadata defining the component versions and outputs.
            pretrained_backbone: Whether to start from ImageNet backbone weights.
            contract_validation: Validate tensors on the first forward pass only, or on
                every forward pass for development and tests.

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
        if manifest.backbone.feature_contract_definition is None:
            raise ValueError("The model requires a resolved, explicit feature contract")
        if contract_validation not in {"first_forward", "every_forward"}:
            raise ValueError(f"Unsupported contract validation mode: {contract_validation}")
        self.manifest = manifest
        self.feature_contract = manifest.backbone.feature_contract_definition
        self.contract_validation = contract_validation
        self._feature_contract_validated = False
        self._validated_head_contracts: set[str] = set()
        self.backbone = ResNet34Backbone(pretrained=pretrained_backbone)
        self.classification = ClassificationHead(manifest.classification.num_classes)
        self.segmentation = SegmentationHead(manifest.segmentation.num_classes)
        for task in ("classification", "segmentation"):
            head_manifest = manifest.head(task)
            contract = head_manifest.interface_contract_definition
            if contract is None:
                raise ValueError(f"The {task} head requires a resolved interface contract")
            implementation = getattr(self, task)
            if contract.consumed_keys != implementation.consumed_feature_keys:
                raise ValueError(
                    f"{task} contract consumes {contract.consumed_keys}, but the implementation "
                    f"consumes {implementation.consumed_feature_keys}"
                )

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
        model_device = next(self.backbone.parameters()).device
        if image.device != model_device:
            raise ValueError(f"Input is on {image.device}, but model is on {model_device}")
        features = self.backbone(image)
        if (
            self.contract_validation == "every_forward"
            or not self._feature_contract_validated
        ):
            self.feature_contract.validate_tensors(image, features)
            self._feature_contract_validated = True
        if task == "classification":
            output = self.classification(features)
        elif task == "segmentation":
            output = self.segmentation(features, image.shape[-2:])
        else:
            raise ValueError(f"Unsupported task: {task}")
        if (
            self.contract_validation == "every_forward"
            or task not in self._validated_head_contracts
        ):
            contract = self.manifest.head(task).interface_contract_definition
            if contract is None:  # Guarded during construction; retained for type narrowing.
                raise RuntimeError(f"Missing resolved {task} head contract")
            contract.validate_tensors(features, output, tuple(image.shape))
            self._validated_head_contracts.add(task)
        return output
