"""Deterministic seeding for reproducible runs.

Genomics evaluation involves random train/test splits (Milestone 4) and stochastic
plotting jitter; pinning the seed means results are reproducible run-to-run. The
torch portion is guarded so this works on the Mac (no torch installed locally) and
on the Modal GPU side (torch present).
"""

from __future__ import annotations

import os
import random

from gvep.config import SEED


def set_seed(seed: int = SEED) -> int:
    """Seed Python, NumPy, and (if available) PyTorch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # No torch locally (Apple Silicon) — fine; the GPU side imports it.
        pass

    return seed
