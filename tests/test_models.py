from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pet.models import PetModel, load_model_manifest
from pet.training.checkpoints import load_checkpoint, save_checkpoint


@pytest.fixture(scope="module")
def manifest():
    return load_model_manifest(Path("configs/model/resnet34_v1.yaml"))


def test_backbone_and_heads_follow_shape_contract(manifest) -> None:
    model = PetModel(manifest)
    image = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        classification = model(image, "classification")
        segmentation = model(image, "segmentation")
    assert classification.shape == (2, 37)
    assert segmentation.shape == (2, 2, 32, 32)


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
