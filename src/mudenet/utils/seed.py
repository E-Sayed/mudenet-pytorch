"""Reproducibility utilities for MuDeNet.

Provides a centralized seed function that configures all RNG sources
and cuDNN determinism settings.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random number generators for reproducibility.

    Configures Python ``random``, NumPy, PyTorch CPU, PyTorch CUDA (all devices),
    and cuDNN determinism settings.

    Args:
        seed: Random seed value.
        deterministic: If True, enables ``torch.backends.cudnn.deterministic``
            and disables ``torch.backends.cudnn.benchmark`` for fully
            reproducible results (at a potential performance cost).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    logger.info("Set seed=%d, deterministic=%s", seed, deterministic)
