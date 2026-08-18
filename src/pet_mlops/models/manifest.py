from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

TaskName = Literal["classification", "segmentation"]


@dataclass(frozen=True)
class BackboneManifest:
    """Identity and output contract for a versioned backbone."""

    name: str
    version: str
    feature_contract: str


@dataclass(frozen=True)
class HeadManifest:
    """Identity, compatibility requirement, and output size for a task head."""

    version: str
    requires_feature_contract: str
    num_classes: int


@dataclass(frozen=True)
class ModelManifest:
    """Versioned compatibility metadata for the backbone and both task heads."""

    schema_version: int
    backbone: BackboneManifest
    classification: HeadManifest
    segmentation: HeadManifest

    def head(self, task: TaskName) -> HeadManifest:
        """Return the manifest entry belonging to a task.

        Args:
            task: Task whose head metadata is requested.

        Returns:
            Classification or segmentation head metadata.

        Raises:
            AttributeError: If ``task`` does not identify a manifest field.
        """
        return getattr(self, task)

    def validate(self, task: TaskName | None = None) -> None:
        """Validate schema, supported backbone, and feature compatibility.

        Args:
            task: Optional single head to validate. When omitted, validates both heads.

        Raises:
            ValueError: If the schema, backbone, class count, or feature contract is invalid.
        """
        if self.schema_version != 1:
            raise ValueError(f"Unsupported model manifest schema_version: {self.schema_version}")
        if self.backbone.name != "resnet34":
            raise ValueError(f"Unsupported backbone: {self.backbone.name}")
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest using its stable YAML/checkpoint structure.

        Returns:
            Plain dictionary containing the schema, backbone, and nested head entries.
        """
        raw = asdict(self)
        return {
            "schema_version": raw["schema_version"],
            "backbone": raw["backbone"],
            "heads": {
                "classification": raw["classification"],
                "segmentation": raw["segmentation"],
            },
        }


def load_model_manifest(path: str | Path) -> ModelManifest:
    """Load and validate model compatibility metadata from YAML.

    Args:
        path: YAML manifest location.

    Returns:
        Validated immutable model manifest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML root is not a mapping.
        KeyError: If a required manifest entry is absent.
        ValueError: If a value violates the model compatibility contract.
        yaml.YAMLError: If the file is not valid YAML.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Model manifest must be a YAML mapping")
    heads = raw["heads"]
    manifest = ModelManifest(
        schema_version=int(raw["schema_version"]),
        backbone=BackboneManifest(**raw["backbone"]),
        classification=HeadManifest(**heads["classification"]),
        segmentation=HeadManifest(**heads["segmentation"]),
    )
    manifest.validate()
    return manifest


def manifests_compatible(
    expected: ModelManifest, actual: ModelManifest, task: TaskName
) -> None:
    """Ensure a checkpoint manifest can supply one requested task.

    The unselected head is intentionally ignored so its version can evolve independently.

    Args:
        expected: Manifest required by the current run.
        actual: Manifest stored with the checkpoint.
        task: Head that will be loaded from the checkpoint.

    Raises:
        ValueError: If the backbone or selected head metadata differs.
    """
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
    mismatches = [
        f"{name}: expected {left!r}, found {right!r}"
        for name, (left, right) in fields.items()
        if left != right
    ]
    if mismatches:
        raise ValueError("Incompatible checkpoint manifest: " + "; ".join(mismatches))
