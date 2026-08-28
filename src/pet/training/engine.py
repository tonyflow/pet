from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from pet.models.manifest import TaskName


def task_loss(logits: Tensor, target: Tensor, task: TaskName) -> Tensor:
    """Compute cross-entropy loss using task-appropriate tensor shapes.
    
    Cross-entropy measures how strongly the model favors the correct class.

    Other plausible choices were:
    1. Binary cross-entropy: possible for binary segmentation using one output channel, 
    but our head deliberately produces two class logits.
    2. Dice loss: useful for segmentation and class imbalance, but less straightforward 
    as a common baseline.
    3. Combined cross-entropy + Dice: often a good later experiment.
    4. Focal loss: useful when difficult or minority examples are being ignored, 
    but adds complexity before we have evidence it is needed.
    
    B = batch size
    C = number of classes/channels
    H = height
    W = width

    Args:
        logits: Unnormalized model scores. Classification uses ``[B, C]`` and
            segmentation uses ``[B, C, H, W]``.
        target: Integer labels shaped ``[B]`` or pixel labels shaped ``[B, H, W]``.
        task: Task determining how the tensors are interpreted.

    Returns:
        Scalar mean cross-entropy loss.

    Raises:
        ValueError: If ``task`` is unsupported.
        RuntimeError: If logits and target shapes are incompatible.
    """
    if task == "classification":
        return nn.functional.cross_entropy(logits, target)
    if task == "segmentation":
        return nn.functional.cross_entropy(logits, target.long())
    raise ValueError(f"Unsupported task: {task}")


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor, object]],
    optimizer: torch.optim.Optimizer,
    task: TaskName,
    device: torch.device,
    *,
    amp: bool = True,
) -> float:
    """Update a model for one complete pass over a task's batches.

    An epoch is one pass over the supplied loader, not the total training run. A higher-level
    runner calls this function once per configured epoch. Mixed precision is enabled only when
    requested on a CUDA device; CPU training remains full precision.

    Args:
        model: Model accepting ``(images, task)`` and returning task logits.
        batches: Iterable yielding images, targets, and sample identifiers.
        optimizer: Optimizer responsible for applying parameter updates.
        task: Task whose head and loss shapes should be used.
        device: Device on which images, targets, and the model reside.
        amp: Whether to request automatic mixed precision when CUDA is used.

    Returns:
        Sample-weighted mean training loss across the epoch.

    Raises:
        ValueError: If the loader is empty or the task is unsupported.
        RuntimeError: If model, data, or device tensors are incompatible.
    """

    # Enter the model in training mode
    model.train()


    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    total_samples = 0
    for images, targets, _sample_ids in batches:

        # Move images and targets to the selected decice
        images = images.to(device)
        targets = targets.to(device)

        # Clear gradients left from the previous run
        optimizer.zero_grad(set_to_none=True)

        # Configure and use autocast
        # TODO what is autocat? why are we doing this?
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with autocast:
            # TODO is this the foward pass
            logits = model(images, task)
            loss = task_loss(logits, targets, task)

        # Calculate gradients
        # TODO what is a scaler and why are we invoking backward on the tensor it returns?
        scaler.scale(loss).backward()

        # Update trainable parameters
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Cannot train on an empty loader")

    # Return the average training loss for that epoch
    return total_loss / total_samples


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect parameters that the optimizer is currently allowed to update.

    Args:
        model: Model whose parameters should be filtered.

    Returns:
        Parameters with ``requires_grad`` enabled.
    """
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
