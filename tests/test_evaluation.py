from __future__ import annotations

import torch
from torch import nn

from pet.training.evaluation import evaluate


class FixedClassificationModel(nn.Module):
    def forward(self, images: torch.Tensor, task: str) -> torch.Tensor:
        assert task == "classification"
        return images[:, :2, 0, 0]


class FixedSegmentationModel(nn.Module):
    def forward(self, images: torch.Tensor, task: str) -> torch.Tensor:
        assert task == "segmentation"
        return torch.stack((-images[:, 0], images[:, 0]), dim=1)


def test_classification_evaluation_retains_predictions() -> None:
    images = torch.tensor([[[[2.0]], [[1.0]]], [[[0.0]], [[3.0]]]])
    targets = torch.tensor([0, 1])

    result = evaluate(
        FixedClassificationModel(),
        [(images, targets, ["cat", "dog"])],
        "classification",
        torch.device("cpu"),
    )

    assert result.metrics["accuracy"] == 1.0
    assert [row["sample_id"] for row in result.predictions] == ["cat", "dog"]


def test_segmentation_evaluation_reports_iou() -> None:
    foreground = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    images = torch.stack((foreground, foreground, foreground), dim=1)
    targets = torch.tensor([[[1, 0], [0, 1]]])

    result = evaluate(
        FixedSegmentationModel(),
        [(images, targets, ["pet"])],
        "segmentation",
        torch.device("cpu"),
    )

    assert result.metrics["pixel_accuracy"] == 1.0
    assert result.metrics["mean_iou"] == 1.0
    assert result.predictions[0]["foreground_iou"] == 1.0
