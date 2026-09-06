from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pet.models import PetModel, load_model_manifest
from pet.models.manifest import contract_fingerprint, manifests_compatible, model_manifest_from_dict
from pet.training.checkpoints import load_checkpoint, save_checkpoint


@pytest.fixture(scope="module")
def manifest():
    return load_model_manifest("resnet34-v1")


def test_backbone_and_heads_follow_shape_contract(manifest) -> None:
    model = PetModel(manifest)
    image = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        classification = model(image, "classification")
        segmentation = model(image, "segmentation")
    assert classification.shape == (2, 37)
    assert segmentation.shape == (2, 2, 32, 32)


def test_contract_is_validated_only_on_first_forward_by_default(manifest) -> None:
    model = PetModel(manifest).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        model(image, "classification")

    original_forward = model.backbone.forward

    def forward_with_extra_output(value):
        features = original_forward(value)
        features["unexpected"] = features["stem"]
        return features

    model.backbone.forward = forward_with_extra_output
    with torch.no_grad():
        assert model(image, "classification").shape == (1, 37)


def test_contract_can_be_validated_on_every_forward(manifest) -> None:
    model = PetModel(manifest, contract_validation="every_forward").eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        model(image, "classification")

    original_forward = model.backbone.forward

    def forward_with_extra_output(value):
        features = original_forward(value)
        features["unexpected"] = features["stem"]
        return features

    model.backbone.forward = forward_with_extra_output
    with pytest.raises(ValueError, match="unexpected"):
        with torch.no_grad():
            model(image, "classification")


def test_contract_validation_mode_is_checked(manifest) -> None:
    with pytest.raises(ValueError, match="Unsupported contract validation mode"):
        PetModel(manifest, contract_validation="sometimes")


def test_manifest_resolves_explicit_feature_contract(manifest) -> None:
    contract = manifest.backbone.feature_contract_definition
    assert contract is not None
    assert contract.id == "resnet34-pyramid-v1"
    assert contract.kind == "backbone-feature-pyramid"
    assert contract.supported_dtypes == ("float16", "bfloat16", "float32")
    assert contract.normalization_mean == (0.485, 0.456, 0.406)
    assert contract.normalization_std == (0.229, 0.224, 0.225)
    assert len(contract.examples) == 2
    assert {
        name: (spec.channels, spec.spatial_stride)
        for name, spec in contract.outputs.items()
    } == {
        "stem": (64, 2),
        "layer1": (64, 4),
        "layer2": (128, 8),
        "layer3": (256, 16),
        "layer4": (512, 32),
    }


def test_manifest_resolves_explicit_head_contracts(manifest) -> None:
    classification = manifest.classification.interface_contract_definition
    segmentation = manifest.segmentation.interface_contract_definition
    assert classification is not None
    assert classification.consumed_keys == ("layer4",)
    assert classification.output_layout == "NC"
    assert classification.num_classes == 37
    assert segmentation is not None
    assert segmentation.consumed_keys == ("stem", "layer1", "layer2", "layer3", "layer4")
    assert segmentation.output_layout == "NCHW"
    assert segmentation.spatial_policy == "same_as_original_image"


def test_head_contract_rejects_wrong_output_shape(manifest) -> None:
    contract = manifest.classification.interface_contract_definition
    feature_contract = manifest.backbone.feature_contract_definition
    assert contract is not None
    assert feature_contract is not None
    features = {
        name: torch.randn(2, spec.channels, 32 // spec.spatial_stride, 32 // spec.spatial_stride)
        for name, spec in feature_contract.outputs.items()
    }
    with pytest.raises(ValueError, match="expected output shape"):
        contract.validate_tensors(features, torch.randn(2, 36), (2, 3, 32, 32))


def test_contract_rejects_incompatible_preprocessing(manifest) -> None:
    contract = manifest.backbone.feature_contract_definition
    assert contract is not None
    with pytest.raises(ValueError, match="normalization does not match"):
        contract.validate_preprocessing((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_contract_fingerprint_is_stable_and_content_sensitive(manifest) -> None:
    contract = manifest.backbone.feature_contract_definition
    assert contract is not None
    assert contract_fingerprint(contract) == manifest.backbone.feature_contract_sha256
    changed = replace(contract, minimum_height=64)
    assert contract_fingerprint(changed) != contract_fingerprint(contract)


def test_unknown_typed_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model manifest"):
        load_model_manifest("missing-v1")


def test_explicit_contract_rejects_wrong_feature_shape(manifest) -> None:
    contract = manifest.backbone.feature_contract_definition
    assert contract is not None
    image = torch.randn(2, 3, 32, 32)
    features = {
        name: torch.randn(2, spec.channels, 32 // spec.spatial_stride, 32 // spec.spatial_stride)
        for name, spec in contract.outputs.items()
    }
    features["layer4"] = torch.randn(2, 256, 1, 1)

    with pytest.raises(ValueError, match=r"layer4.*expected shape"):
        contract.validate_tensors(image, features)


def test_training_modes_control_only_backbone(manifest) -> None:
    model = PetModel(manifest)
    model.set_train_mode("frozen_backbone")
    model.train()
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classification.parameters())

    model.set_train_mode("fine_tune")
    model.train()
    assert model.backbone.training
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_manifest_rejects_incompatible_feature_contract(manifest) -> None:
    incompatible_head = replace(
        manifest.classification, requires_feature_contract="some-other-contract"
    )
    with pytest.raises(ValueError, match="requires feature contract"):
        replace(manifest, classification=incompatible_head).validate("classification")


def test_checkpoint_round_trip_and_version_validation(tmp_path: Path, manifest) -> None:
    source = PetModel(manifest)
    checkpoint = tmp_path / "classification.pt"
    save_checkpoint(checkpoint, source, manifest, "classification", epoch=3)

    restored = PetModel(manifest)
    assert load_checkpoint(checkpoint, restored, manifest, "classification") == 3
    source_value = next(source.classification.parameters()).detach()
    restored_value = next(restored.classification.parameters()).detach()
    assert torch.equal(source_value, restored_value)

    incompatible = replace(
        manifest,
        classification=replace(manifest.classification, version="classification-head-v2"),
    )
    with pytest.raises(ValueError, match=r"head\.version"):
        load_checkpoint(checkpoint, restored, incompatible, "classification")


def test_head_versions_are_independent_during_checkpoint_load(tmp_path: Path, manifest) -> None:
    model = PetModel(manifest)
    checkpoint = tmp_path / "classification.pt"
    save_checkpoint(checkpoint, model, manifest, "classification", epoch=0)
    changed_other_head = replace(
        manifest,
        segmentation=replace(manifest.segmentation, version="segmentation-head-v2"),
    )
    assert load_checkpoint(checkpoint, model, changed_other_head, "classification") == 0


def test_legacy_string_only_checkpoint_manifest_remains_compatible(manifest) -> None:
    legacy_raw = manifest.to_dict()
    legacy_raw["backbone"].pop("feature_contract_definition")
    legacy_raw["backbone"].pop("feature_contract_sha256")
    for head in legacy_raw["heads"].values():
        head.pop("interface_contract")
        head.pop("interface_contract_definition")
        head.pop("interface_contract_sha256")
    legacy = model_manifest_from_dict(legacy_raw)

    manifests_compatible(manifest, legacy, "classification")
