from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn

from pet_mlops.models.manifest import TaskName
from pet_mlops.training.engine import task_loss


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics and stable per-sample predictions for one split."""

    metrics: dict[str, float | int]
    predictions: list[dict[str, object]]


def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor, object]],
    task: TaskName,
    device: torch.device,
    *,
    amp: bool = True,
) -> EvaluationResult:
    """Evaluate a task without gradients and retain per-sample predictions."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    correct = 0
    pixels = 0
    intersection = torch.zeros(2, dtype=torch.int64)
    union = torch.zeros(2, dtype=torch.int64)
    predictions: list[dict[str, object]] = []
    started = perf_counter()

    with torch.inference_mode():
        for images, targets, sample_ids in batches:
            images = images.to(device)
            targets = targets.to(device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast:
                logits = model(images, task)
                loss = task_loss(logits, targets, task)
            predicted = logits.argmax(dim=1)
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if task == "classification":
                probabilities = logits.softmax(dim=1)
                correct += int((predicted == targets).sum().item())
                for index, sample_id in enumerate(sample_ids):
                    label = int(predicted[index].item())
                    predictions.append(
                        {
                            "sample_id": str(sample_id),
                            "target": int(targets[index].item()),
                            "prediction": label,
                            "confidence": float(probabilities[index, label].item()),
                        }
                    )
            else:
                correct += int((predicted == targets).sum().item())
                pixels += targets.numel()
                for class_id in range(2):
                    predicted_class = predicted == class_id
                    target_class = targets == class_id
                    intersection[class_id] += (predicted_class & target_class).sum().cpu()
                    union[class_id] += (predicted_class | target_class).sum().cpu()
                for index, sample_id in enumerate(sample_ids):
                    sample_prediction = predicted[index]
                    sample_target = targets[index]
                    foreground_intersection = (
                        (sample_prediction == 1) & (sample_target == 1)
                    ).sum()
                    foreground_union = (
                        (sample_prediction == 1) | (sample_target == 1)
                    ).sum()
                    predictions.append(
                        {
                            "sample_id": str(sample_id),
                            "pixel_accuracy": float(
                                (sample_prediction == sample_target).float().mean().item()
                            ),
                            "foreground_iou": float(
                                foreground_intersection.item() / max(1, foreground_union.item())
                            ),
                        }
                    )

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty loader")
    elapsed = perf_counter() - started
    metrics: dict[str, float | int] = {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "latency_seconds": elapsed,
        "latency_ms_per_sample": elapsed * 1000 / total_samples,
    }
    if task == "classification":
        metrics["accuracy"] = correct / total_samples
    else:
        iou = intersection.float() / union.clamp_min(1)
        metrics.update(
            {
                "pixel_accuracy": correct / pixels,
                "mean_iou": float(iou.mean().item()),
                "foreground_iou": float(iou[1].item()),
            }
        )
    return EvaluationResult(metrics=metrics, predictions=predictions)
