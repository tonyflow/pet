from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

TaskName = Literal["classification", "segmentation"]


def contract_fingerprint(contract: FeatureContract | HeadContract) -> str:
    """Return a stable SHA-256 fingerprint of a canonical contract snapshot."""
    payload = json.dumps(
        contract.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FeatureTensorSpec:
    """Required channel count and spatial stride for one backbone output."""

    channels: int
    spatial_stride: int
    source: str
    semantic_level: str


@dataclass(frozen=True)
class FeatureContractExample:
    """Concrete input and output shapes illustrating a feature contract."""

    input_shape: tuple[int, int, int, int]
    outputs: Mapping[str, tuple[int, int, int, int]]


@dataclass(frozen=True)
class FeatureContract:
    """Resolved, explicit tensor interface provided by a backbone."""

    schema_version: int
    id: str
    kind: str
    description: str
    input_name: str
    input_channels: int
    input_layout: str
    input_rank: int
    minimum_height: int
    minimum_width: int
    supported_dtypes: tuple[str, ...]
    color_space: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    additional_keys_allowed: bool
    outputs: Mapping[str, FeatureTensorSpec]
    deterministic_given_deterministic_runtime: bool
    examples: tuple[FeatureContractExample, ...]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported feature contract schema_version: {self.schema_version}"
            )
        if self.input_layout != "NCHW" or self.input_rank != 4:
            raise ValueError("Only rank-4 NCHW backbone inputs are supported")
        if self.kind != "backbone-feature-pyramid" or self.input_name != "image":
            raise ValueError("Unsupported feature contract kind or input name")
        if self.input_channels < 1 or self.minimum_height < 1 or self.minimum_width < 1:
            raise ValueError("Feature contract input dimensions must be positive")
        if not self.outputs:
            raise ValueError("Feature contract must define at least one output")
        supported = {"float16", "bfloat16", "float32"}
        if not self.supported_dtypes or not set(self.supported_dtypes) <= supported:
            raise ValueError(f"Unsupported input dtypes: {self.supported_dtypes}")
        if self.color_space != "RGB":
            raise ValueError(f"Unsupported input color space: {self.color_space}")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("Normalization mean and std must each contain 3 values")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("Normalization standard deviations must be positive")
        for name, spec in self.outputs.items():
            if not name or spec.channels < 1 or spec.spatial_stride < 1:
                raise ValueError(f"Invalid feature output specification: {name!r}")
        for example in self.examples:
            self._validate_example(example)

    def validate_preprocessing(
        self, mean: tuple[float, ...], std: tuple[float, ...]
    ) -> None:
        """Ensure a data pipeline uses the normalization declared by the contract."""
        if mean != self.normalization_mean or std != self.normalization_std:
            raise ValueError(
                "Data normalization does not match the backbone feature contract: "
                f"expected mean={self.normalization_mean}, std={self.normalization_std}; "
                f"found mean={mean}, std={std}"
            )

    def _expected_shape(
        self, batch: int, height: int, width: int, spec: FeatureTensorSpec
    ) -> tuple[int, int, int, int]:
        return (
            batch,
            spec.channels,
            (height + spec.spatial_stride - 1) // spec.spatial_stride,
            (width + spec.spatial_stride - 1) // spec.spatial_stride,
        )

    def _validate_example(self, example: FeatureContractExample) -> None:
        batch, channels, height, width = example.input_shape
        if channels != self.input_channels or batch < 1:
            raise ValueError(f"Invalid contract example input: {example.input_shape}")
        if set(example.outputs) != set(self.outputs):
            raise ValueError("Contract example output keys do not match required outputs")
        for name, spec in self.outputs.items():
            expected = self._expected_shape(batch, height, width, spec)
            if example.outputs[name] != expected:
                raise ValueError(
                    f"Contract example {name!r} expected {expected}, "
                    f"found {example.outputs[name]}"
                )

    def validate_tensors(self, image: Tensor, features: Mapping[str, Tensor]) -> None:
        """Validate actual backbone input and outputs against this contract."""
        if image.ndim != self.input_rank:
            raise ValueError(f"Expected input rank {self.input_rank}, found {image.ndim}")
        batch, channels, height, width = image.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, found {channels}")
        if height < self.minimum_height or width < self.minimum_width:
            raise ValueError(
                f"Input spatial size must be at least {self.minimum_height}x"
                f"{self.minimum_width}, found {height}x{width}"
            )
        dtype_name = str(image.dtype).removeprefix("torch.")
        if dtype_name not in self.supported_dtypes:
            raise ValueError(
                f"Input dtype {dtype_name!r} is not supported; expected {self.supported_dtypes}"
            )
        expected_keys = set(self.outputs)
        actual_keys = set(features)
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        if missing or (unexpected and not self.additional_keys_allowed):
            raise ValueError(
                f"Feature keys do not satisfy {self.id}: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        for name, spec in self.outputs.items():
            tensor = features[name]
            expected_shape = self._expected_shape(batch, height, width, spec)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Feature {name!r} expected shape {expected_shape}, "
                    f"found {tuple(tensor.shape)}"
                )
            if tensor.device != image.device:
                raise ValueError(
                    f"Feature {name!r} is on {tensor.device}, input is on {image.device}"
                )
            if not torch.is_floating_point(tensor):
                raise ValueError(f"Feature {name!r} must use a floating-point dtype")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": {
                "id": self.id,
                "kind": self.kind,
                "description": self.description,
                "additional_outputs_allowed": self.additional_keys_allowed,
            },
            "input": {
                "name": self.input_name,
                "container": "torch.Tensor",
                "layout": self.input_layout,
                "rank": self.input_rank,
                "dimensions": {
                    "batch": {"symbol": "B", "minimum": 1},
                    "channels": {"exact": self.input_channels},
                    "height": {"symbol": "H", "minimum": self.minimum_height},
                    "width": {"symbol": "W", "minimum": self.minimum_width},
                },
                "dtype": {
                    "category": "floating_point",
                    "supported": list(self.supported_dtypes),
                    "policy": "follows_model_compute_dtype",
                },
                "device": {"policy": "same_device_as_model"},
                "value_semantics": {
                    "color_space": self.color_space,
                    "range": "normalized",
                    "normalization": {
                        "mean": list(self.normalization_mean),
                        "std": list(self.normalization_std),
                    },
                },
            },
            "output": {
                "container": "mapping",
                "key_type": "string",
                "key_order_significant": False,
                "required_keys": list(self.outputs),
                "shared_properties": {
                    "batch_dimension": "same_as_input",
                    "layout": "NCHW",
                    "dtype": "follows_model_compute_dtype",
                    "device": "same_as_input",
                    "differentiable": True,
                    "contiguous_required": False,
                },
                "tensors": {
                    name: {
                        "container": "torch.Tensor",
                        "rank": 4,
                        "channels": spec.channels,
                        "spatial_stride": spec.spatial_stride,
                        "source": spec.source,
                        "semantic_level": spec.semantic_level,
                    }
                    for name, spec in self.outputs.items()
                },
            },
            "behavior": {
                "preserves_batch_size": True,
                "accepts_dynamic_batch_size": True,
                "accepts_dynamic_spatial_size": True,
                "deterministic_given_deterministic_runtime": (
                    self.deterministic_given_deterministic_runtime
                ),
                "mutates_input": False,
            },
            "examples": [
                {
                    "input_shape": list(example.input_shape),
                    "outputs": {
                        name: list(shape) for name, shape in example.outputs.items()
                    },
                }
                for example in self.examples
            ],
        }


@dataclass(frozen=True)
class BackboneManifest:
    """Identity and resolved output contract for a versioned backbone."""

    name: str
    version: str
    feature_contract: str
    feature_contract_definition: FeatureContract | None = None
    feature_contract_sha256: str | None = None


@dataclass(frozen=True)
class HeadContract:
    """Explicit feature-input and logit-output interface for one task head."""

    schema_version: int
    id: str
    kind: str
    description: str
    feature_contract: str
    consumed_keys: tuple[str, ...]
    output_name: str
    output_rank: int
    output_layout: str
    num_classes: int
    spatial_policy: str | None
    semantics: str

    def validate(self, task: TaskName) -> None:
        if self.schema_version != 1 or self.kind != f"{task}-head":
            raise ValueError(f"Invalid {task} head contract schema or kind")
        expected_rank = 2 if task == "classification" else 4
        expected_layout = "NC" if task == "classification" else "NCHW"
        if self.output_rank != expected_rank or self.output_layout != expected_layout:
            raise ValueError(f"Invalid {task} head output rank or layout")
        if not self.consumed_keys or self.num_classes < 2:
            raise ValueError(f"Invalid {task} head feature keys or class count")
        if task == "segmentation" and self.spatial_policy != "same_as_original_image":
            raise ValueError("Segmentation output must match the original image size")

    def validate_tensors(
        self,
        features: Mapping[str, Tensor],
        output: Tensor,
        image_shape: tuple[int, ...],
    ) -> None:
        missing = set(self.consumed_keys) - set(features)
        if missing:
            raise ValueError(f"Head contract {self.id} is missing features: {sorted(missing)}")
        expected = (image_shape[0], self.num_classes)
        if self.output_rank == 4:
            expected += (image_shape[-2], image_shape[-1])
        if tuple(output.shape) != expected:
            raise ValueError(
                f"Head contract {self.id} expected output shape {expected}, "
                f"found {tuple(output.shape)}"
            )
        reference = features[self.consumed_keys[0]]
        if output.device != reference.device:
            raise ValueError(
                f"Head output is on {output.device}, but its features are on {reference.device}"
            )
        if not torch.is_floating_point(output):
            raise ValueError("Head output must use a floating-point dtype")

    def to_dict(self) -> dict[str, Any]:
        dimensions: dict[str, Any] = {
            "batch": "same_as_input",
            "classes": self.num_classes,
        }
        input_spec: dict[str, Any] = {
            "container": "mapping",
            "feature_contract": self.feature_contract,
            "consumed_keys": list(self.consumed_keys),
        }
        if self.spatial_policy is not None:
            input_spec["context"] = {
                "original_image_size": {
                    "dimensions": ["height", "width"],
                    "source": "model_input",
                }
            }
            dimensions["height"] = self.spatial_policy
            dimensions["width"] = self.spatial_policy
        return {
            "schema_version": self.schema_version,
            "contract": {
                "id": self.id,
                "kind": self.kind,
                "description": self.description,
            },
            "input": input_spec,
            "output": {
                "name": self.output_name,
                "container": "torch.Tensor",
                "rank": self.output_rank,
                "layout": self.output_layout,
                "dimensions": dimensions,
                "dtype": "follows_model_compute_dtype",
                "device": "same_as_input",
                "semantics": self.semantics,
            },
        }


@dataclass(frozen=True)
class HeadManifest:
    """Identity, compatibility requirement, and output size for a task head."""

    version: str
    requires_feature_contract: str
    num_classes: int
    interface_contract: str | None = None
    interface_contract_definition: HeadContract | None = None
    interface_contract_sha256: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    """Versioned compatibility metadata for the backbone and both task heads."""

    schema_version: int
    backbone: BackboneManifest
    classification: HeadManifest
    segmentation: HeadManifest

    def head(self, task: TaskName) -> HeadManifest:
        return getattr(self, task)

    def validate(self, task: TaskName | None = None) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported model manifest schema_version: {self.schema_version}")
        if self.backbone.name != "resnet34":
            raise ValueError(f"Unsupported backbone: {self.backbone.name}")
        contract = self.backbone.feature_contract_definition
        if contract is not None:
            contract.validate()
            if contract.id != self.backbone.feature_contract:
                raise ValueError(
                    f"Feature contract reference {self.backbone.feature_contract!r} does not "
                    f"match definition {contract.id!r}"
                )
            if (
                self.backbone.feature_contract_sha256 is not None
                and contract_fingerprint(contract) != self.backbone.feature_contract_sha256
            ):
                raise ValueError("Backbone feature contract fingerprint does not match")
        tasks = (task,) if task else ("classification", "segmentation")
        for name in tasks:
            head = self.head(name)
            if head.requires_feature_contract != self.backbone.feature_contract:
                raise ValueError(
                    f"{name} head requires feature contract "
                    f"{head.requires_feature_contract!r}, but backbone provides "
                    f"{self.backbone.feature_contract!r}"
                )
            if head.num_classes < 2:
                raise ValueError(f"{name} num_classes must be at least 2")
            head_contract = head.interface_contract_definition
            if head_contract is not None:
                head_contract.validate(name)
                if head.interface_contract != head_contract.id:
                    raise ValueError(f"{name} head contract ID does not match its definition")
                if head_contract.feature_contract != head.requires_feature_contract:
                    raise ValueError(f"{name} head contract requires a different feature contract")
                if head_contract.num_classes != head.num_classes:
                    raise ValueError(
                        f"{name} head contract class count does not match its manifest"
                    )
                if (
                    head.interface_contract_sha256 is not None
                    and contract_fingerprint(head_contract) != head.interface_contract_sha256
                ):
                    raise ValueError(f"{name} head contract fingerprint does not match")

    def to_dict(self) -> dict[str, Any]:
        backbone: dict[str, Any] = {
            "name": self.backbone.name,
            "version": self.backbone.version,
            "feature_contract": self.backbone.feature_contract,
        }
        if self.backbone.feature_contract_definition is not None:
            backbone["feature_contract_definition"] = (
                self.backbone.feature_contract_definition.to_dict()
            )
        if self.backbone.feature_contract_sha256 is not None:
            backbone["feature_contract_sha256"] = self.backbone.feature_contract_sha256
        heads: dict[str, dict[str, Any]] = {}
        for task in ("classification", "segmentation"):
            head = self.head(task)
            serialized: dict[str, Any] = {
                "version": head.version,
                "requires_feature_contract": head.requires_feature_contract,
                "num_classes": head.num_classes,
            }
            if head.interface_contract is not None:
                serialized["interface_contract"] = head.interface_contract
            if head.interface_contract_definition is not None:
                serialized["interface_contract_definition"] = (
                    head.interface_contract_definition.to_dict()
                )
            if head.interface_contract_sha256 is not None:
                serialized["interface_contract_sha256"] = head.interface_contract_sha256
            heads[task] = serialized
        return {
            "schema_version": self.schema_version,
            "backbone": backbone,
            "heads": heads,
        }


def _load_contract(raw: Mapping[str, Any]) -> FeatureContract:
    if "contract" not in raw:
        return _load_legacy_contract(raw)

    contract_spec = raw["contract"]
    input_spec = raw["input"]
    dimensions = input_spec["dimensions"]
    dtype_spec = input_spec["dtype"]
    value_semantics = input_spec["value_semantics"]
    normalization = value_semantics["normalization"]
    output_spec = raw["output"]
    shared_output = output_spec["shared_properties"]
    behavior = raw["behavior"]
    expected_input_literals = {
        "container": "torch.Tensor",
        "layout": "NCHW",
        "rank": 4,
    }
    expected_output_literals = {
        "batch_dimension": "same_as_input",
        "layout": "NCHW",
        "dtype": "follows_model_compute_dtype",
        "device": "same_as_input",
        "differentiable": True,
        "contiguous_required": False,
    }
    for field, expected in expected_input_literals.items():
        if input_spec.get(field) != expected:
            raise ValueError(
                f"Unsupported feature contract input {field}: {input_spec.get(field)!r}"
            )
    for field, expected in expected_output_literals.items():
        if shared_output.get(field) != expected:
            raise ValueError(
                f"Unsupported shared output property {field}: {shared_output.get(field)!r}"
            )
    if dtype_spec.get("category") != "floating_point":
        raise ValueError("Feature contract input dtype must be floating_point")
    if dtype_spec.get("policy") != "follows_model_compute_dtype":
        raise ValueError("Unsupported feature contract input dtype policy")
    if input_spec.get("device", {}).get("policy") != "same_device_as_model":
        raise ValueError("Unsupported feature contract input device policy")
    if output_spec.get("container") != "mapping" or output_spec.get("key_type") != "string":
        raise ValueError("Feature contract output must be a string-keyed mapping")
    if output_spec.get("key_order_significant") is not False:
        raise ValueError("Feature contract output key order must be insignificant")
    expected_behavior = {
        "preserves_batch_size": True,
        "accepts_dynamic_batch_size": True,
        "accepts_dynamic_spatial_size": True,
        "mutates_input": False,
    }
    for field, expected in expected_behavior.items():
        if behavior.get(field) != expected:
            raise ValueError(f"Unsupported feature contract behavior {field}")
    if dimensions["batch"].get("symbol") != "B" or dimensions["batch"].get("minimum") != 1:
        raise ValueError("Feature contract batch dimension must be dynamic and at least 1")
    if dimensions["height"].get("symbol") != "H" or dimensions["width"].get("symbol") != "W":
        raise ValueError("Feature contract spatial dimensions must use H and W symbols")
    if value_semantics.get("range") != "normalized":
        raise ValueError("Feature contract input range must be normalized")

    outputs = {
        str(name): FeatureTensorSpec(
            channels=int(spec["channels"]),
            spatial_stride=int(spec["spatial_stride"]),
            source=str(spec["source"]),
            semantic_level=str(spec["semantic_level"]),
        )
        for name, spec in output_spec["tensors"].items()
    }
    required_keys = [str(name) for name in output_spec["required_keys"]]
    if required_keys != list(outputs):
        raise ValueError("Feature contract required_keys must match tensor definitions in order")
    for name, spec in output_spec["tensors"].items():
        if spec.get("container") != "torch.Tensor" or spec.get("rank") != 4:
            raise ValueError(f"Feature output {name!r} must be a rank-4 torch.Tensor")

    contract = FeatureContract(
        schema_version=int(raw["schema_version"]),
        id=str(contract_spec["id"]),
        kind=str(contract_spec["kind"]),
        description=str(contract_spec["description"]),
        input_name=str(input_spec["name"]),
        input_channels=int(dimensions["channels"]["exact"]),
        input_layout=str(input_spec["layout"]),
        input_rank=int(input_spec["rank"]),
        minimum_height=int(dimensions["height"]["minimum"]),
        minimum_width=int(dimensions["width"]["minimum"]),
        supported_dtypes=tuple(str(value) for value in dtype_spec["supported"]),
        color_space=str(value_semantics["color_space"]),
        normalization_mean=tuple(float(value) for value in normalization["mean"]),
        normalization_std=tuple(float(value) for value in normalization["std"]),
        additional_keys_allowed=bool(contract_spec["additional_outputs_allowed"]),
        outputs=outputs,
        deterministic_given_deterministic_runtime=bool(
            behavior["deterministic_given_deterministic_runtime"]
        ),
        examples=tuple(
            FeatureContractExample(
                input_shape=tuple(int(value) for value in example["input_shape"]),
                outputs={
                    str(name): tuple(int(value) for value in shape)
                    for name, shape in example["outputs"].items()
                },
            )
            for example in raw["examples"]
        ),
    )
    contract.validate()
    return contract


def _load_legacy_contract(raw: Mapping[str, Any]) -> FeatureContract:
    """Load contracts embedded by the earlier schema-1 serializer."""
    input_spec = raw["input"]
    output_spec = raw["outputs"]
    source_names = {
        "stem": ("conv1_batchnorm_relu", "low_level_edges_and_textures"),
        "layer1": ("resnet_layer1", "low_level_features"),
        "layer2": ("resnet_layer2", "intermediate_features"),
        "layer3": ("resnet_layer3", "high_level_features"),
        "layer4": ("resnet_layer4", "deepest_semantic_features"),
    }
    contract = FeatureContract(
        schema_version=int(raw["schema_version"]),
        id=str(raw["id"]),
        kind="backbone-feature-pyramid",
        description="ResNet-34 feature pyramid before global pooling",
        input_name="image",
        input_channels=int(input_spec["channels"]),
        input_layout=str(input_spec["layout"]),
        input_rank=int(input_spec["rank"]),
        minimum_height=int(input_spec["minimum_height"]),
        minimum_width=int(input_spec["minimum_width"]),
        supported_dtypes=("float16", "bfloat16", "float32"),
        color_space="RGB",
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        additional_keys_allowed=bool(output_spec["additional_keys_allowed"]),
        outputs={
            str(name): FeatureTensorSpec(
                channels=int(spec["channels"]),
                spatial_stride=int(spec["spatial_stride"]),
                source=source_names[str(name)][0],
                semantic_level=source_names[str(name)][1],
            )
            for name, spec in output_spec["tensors"].items()
        },
        deterministic_given_deterministic_runtime=True,
        examples=(),
    )
    contract.validate()
    return contract


def _load_head_contract(raw: Mapping[str, Any], task: TaskName) -> HeadContract:
    contract_spec = raw["contract"]
    input_spec = raw["input"]
    output_spec = raw["output"]
    if input_spec.get("container") != "mapping":
        raise ValueError(f"{task} head input must be a feature mapping")
    expected_output = {
        "container": "torch.Tensor",
        "dtype": "follows_model_compute_dtype",
        "device": "same_as_input",
    }
    for field, expected in expected_output.items():
        if output_spec.get(field) != expected:
            raise ValueError(f"Unsupported {task} head output {field}")
    dimensions = output_spec["dimensions"]
    if dimensions.get("batch") != "same_as_input":
        raise ValueError(f"{task} head must preserve the batch dimension")
    spatial_policy = None
    if task == "segmentation":
        if dimensions.get("height") != dimensions.get("width"):
            raise ValueError("Segmentation height and width policies must match")
        spatial_policy = str(dimensions["height"])
        context = input_spec.get("context", {}).get("original_image_size", {})
        if context.get("source") != "model_input":
            raise ValueError("Segmentation original image size must come from the model input")
    contract = HeadContract(
        schema_version=int(raw["schema_version"]),
        id=str(contract_spec["id"]),
        kind=str(contract_spec["kind"]),
        description=str(contract_spec["description"]),
        feature_contract=str(input_spec["feature_contract"]),
        consumed_keys=tuple(str(key) for key in input_spec["consumed_keys"]),
        output_name=str(output_spec["name"]),
        output_rank=int(output_spec["rank"]),
        output_layout=str(output_spec["layout"]),
        num_classes=int(dimensions["classes"]),
        spatial_policy=spatial_policy,
        semantics=str(output_spec["semantics"]),
    )
    contract.validate(task)
    return contract


def _load_head_manifest(
    raw: Mapping[str, Any],
    task: TaskName,
) -> HeadManifest:
    definition = raw.get("interface_contract_definition")
    digest = raw.get("interface_contract_sha256")
    contract: HeadContract | None = None
    if definition is not None:
        contract = _load_head_contract(definition, task)
    return HeadManifest(
        version=str(raw["version"]),
        requires_feature_contract=str(raw["requires_feature_contract"]),
        num_classes=int(raw["num_classes"]),
        interface_contract=(
            str(raw["interface_contract"]) if raw.get("interface_contract") else None
        ),
        interface_contract_definition=contract,
        interface_contract_sha256=(str(digest) if digest else None),
    )


def model_manifest_from_dict(raw: Mapping[str, Any]) -> ModelManifest:
    """Build a model manifest from a canonical checkpoint snapshot."""
    raw_backbone = raw["backbone"]
    contract_definition = raw_backbone.get("feature_contract_definition")
    contract_sha256 = raw_backbone.get("feature_contract_sha256")
    contract: FeatureContract | None = None
    if contract_definition is not None:
        contract = _load_contract(contract_definition)

    heads = raw["heads"]
    manifest = ModelManifest(
        schema_version=int(raw["schema_version"]),
        backbone=BackboneManifest(
            name=str(raw_backbone["name"]),
            version=str(raw_backbone["version"]),
            feature_contract=str(raw_backbone["feature_contract"]),
            feature_contract_definition=contract,
            feature_contract_sha256=(str(contract_sha256) if contract_sha256 else None),
        ),
        classification=_load_head_manifest(heads["classification"], "classification"),
        segmentation=_load_head_manifest(heads["segmentation"], "segmentation"),
    )
    manifest.validate()
    return manifest


def load_model_manifest(manifest_id: str) -> ModelManifest:
    """Resolve and validate a versioned typed model manifest by stable ID."""
    from pet.models.contracts import MODEL_MANIFESTS

    try:
        manifest = MODEL_MANIFESTS[manifest_id]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_MANIFESTS))
        raise ValueError(
            f"Unknown model manifest {manifest_id!r}; available manifests: {available}"
        ) from error
    manifest.validate()
    return manifest


def manifests_compatible(
    expected: ModelManifest, actual: ModelManifest, task: TaskName
) -> None:
    """Ensure a checkpoint manifest can supply one requested task."""
    expected.validate(task)
    actual.validate(task)
    fields = {
        "backbone.name": (expected.backbone.name, actual.backbone.name),
        "backbone.version": (expected.backbone.version, actual.backbone.version),
        "backbone.feature_contract": (
            expected.backbone.feature_contract,
            actual.backbone.feature_contract,
        ),
        "head.version": (expected.head(task).version, actual.head(task).version),
        "head.num_classes": (expected.head(task).num_classes, actual.head(task).num_classes),
    }
    if (
        expected.backbone.feature_contract_definition is not None
        and actual.backbone.feature_contract_definition is not None
    ):
        fields["backbone.feature_contract_definition"] = (
            expected.backbone.feature_contract_definition.to_dict(),
            actual.backbone.feature_contract_definition.to_dict(),
        )
    expected_head_contract = expected.head(task).interface_contract_definition
    actual_head_contract = actual.head(task).interface_contract_definition
    if expected_head_contract is not None and actual_head_contract is not None:
        fields["head.interface_contract_definition"] = (
            expected_head_contract.to_dict(),
            actual_head_contract.to_dict(),
        )
    mismatches = [
        f"{name}: expected {left!r}, found {right!r}"
        for name, (left, right) in fields.items()
        if left != right
    ]
    if mismatches:
        raise ValueError("Incompatible checkpoint manifest: " + "; ".join(mismatches))
