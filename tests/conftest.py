"""Shared test fixtures for mudenet-pytorch."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from mudenet.config.schema import ModelConfig


class DictDataset(Dataset):  # type: ignore[type-arg]
    """Tiny dataset returning {"image": Tensor} dicts.

    Shared across training test modules to avoid duplication.
    """

    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[idx]}


@pytest.fixture
def default_model_config() -> ModelConfig:
    """Default model configuration matching paper defaults."""
    return ModelConfig()


@pytest.fixture
def small_model_config() -> ModelConfig:
    """Reduced-dimension config for fast testing."""
    return ModelConfig(
        num_channels=16,
        latent_dim=8,
        num_levels=2,
        internal_channels=16,
        block_depths=(1, 1),
        kernel_sizes=(3, 3),
        image_size=256,
    )


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    """Small batch of random images (B=2, 3, 256, 256)."""
    return torch.randn(2, 3, 256, 256)


@pytest.fixture
def single_image() -> torch.Tensor:
    """Single random image (B=1, 3, 256, 256)."""
    return torch.randn(1, 3, 256, 256)
