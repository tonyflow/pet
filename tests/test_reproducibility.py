import random

import numpy as np
import torch

from pet_mlops.reproducibility import seed_everything


def test_seed_everything_repeats_rng_sequences() -> None:
    """Reproduce Python, NumPy, and PyTorch random-number sequences."""
    seed_everything(42)
    first = (random.random(), np.random.random(), torch.rand(1))
    seed_everything(42)
    second = (random.random(), np.random.random(), torch.rand(1))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
