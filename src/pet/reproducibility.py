from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    """Seed supported RNGs and configure deterministic PyTorch execution.

    Args:
        seed: Seed shared by Python, NumPy, and PyTorch RNGs.
        deterministic_algorithms: Whether PyTorch must use deterministic operations.

    Returns:
        None.

    Raises:
        RuntimeError: If deterministic mode cannot be configured by the installed PyTorch runtime.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs from a PyTorch DataLoader worker seed.

    Args:
        worker_id: DataLoader worker index supplied by PyTorch.

    Returns:
        None.

    Raises:
        RuntimeError: If called outside a valid PyTorch worker-seeding context.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seeded_generator(seed: int) -> torch.Generator:
    """Create a PyTorch generator initialized with the provided seed.

    Args:
        seed: Initial seed for the generator.

    Returns:
        A deterministically seeded PyTorch generator.

    Raises:
        RuntimeError: If PyTorch cannot create or seed the generator.
    """
    return torch.Generator().manual_seed(seed)
