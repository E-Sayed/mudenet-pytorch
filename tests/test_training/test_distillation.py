"""Tests for the distillation training loop (Stage 1).

These tests use small models and tiny synthetic batches — they verify
that the training mechanics work (parameter updates, checkpoint saving),
not that a full training run produces good results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from mudenet.config.schema import DistillationConfig
from mudenet.models.feature_extractor import FeatureExtractor
from mudenet.models.teacher import TeacherNetwork
from mudenet.training.distillation import _upsample_target, train_distillation
from tests.conftest import DictDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_teacher() -> TeacherNetwork:
    """Small teacher network for fast testing."""
    return TeacherNetwork(
        internal_channels=16,
        output_channels=32,
        num_levels=2,
        block_depths=(1, 1),
        kernel_sizes=(3, 3),
    )


@pytest.fixture
def tiny_dataloader() -> DataLoader:
    """Tiny dataloader with 4 synthetic images."""
    torch.manual_seed(0)
    images = torch.randn(4, 3, 256, 256)
    return DataLoader(DictDataset(images), batch_size=2, shuffle=False)


@pytest.fixture
def distillation_config() -> DistillationConfig:
    """Minimal distillation config for testing (1 epoch)."""
    return DistillationConfig(
        backbone="wide_resnet50_2",
        num_epochs=1,
        learning_rate=1e-3,
        batch_size=2,
    )


@pytest.fixture
def mock_feature_extractor() -> MagicMock:
    """Mock FeatureExtractor that returns random (B, 32, 64, 64) tensors.

    Avoids downloading pretrained WideResNet50 weights in CI.
    The output_channels=32 matches the small_teacher fixture.
    """
    mock_fe = MagicMock(spec=FeatureExtractor)
    mock_fe.training = False
    mock_fe.channel_indices = torch.arange(32)

    def mock_forward(images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        return torch.randn(b, 32, 64, 64)

    mock_fe.side_effect = mock_forward
    mock_fe.to = MagicMock(return_value=mock_fe)
    return mock_fe


# ---------------------------------------------------------------------------
# _upsample_target
# ---------------------------------------------------------------------------


class TestUpsampleTarget:
    """Tests for the _upsample_target helper."""

    def test_upsample_64_to_128(self) -> None:
        """Upsamples (B, C, 64, 64) to (B, C, 128, 128)."""
        x = torch.randn(2, 4, 64, 64)
        result = _upsample_target(x, (128, 128))
        assert result.shape == (2, 4, 128, 128)

    def test_noop_if_already_correct(self) -> None:
        """Returns same tensor if already at target size."""
        x = torch.randn(2, 4, 128, 128)
        result = _upsample_target(x, (128, 128))
        assert result is x  # same object, not a copy

    def test_preserves_batch_and_channels(self) -> None:
        """Batch and channel dims are preserved."""
        x = torch.randn(3, 8, 32, 32)
        result = _upsample_target(x, (128, 128))
        assert result.shape[0] == 3
        assert result.shape[1] == 8


# ---------------------------------------------------------------------------
# train_distillation — integration (small models, 1 epoch)
# ---------------------------------------------------------------------------


class TestTrainDistillation:
    """Integration tests for train_distillation."""

    def test_parameters_change_after_one_epoch(
        self,
        small_teacher: TeacherNetwork,
        tiny_dataloader: DataLoader,
        distillation_config: DistillationConfig,
        mock_feature_extractor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """After 1 epoch of distillation, teacher parameters should change."""
        initial_params = {
            name: param.clone()
            for name, param in small_teacher.named_parameters()
        }

        trained = train_distillation(
            teacher=small_teacher,
            feature_extractor=mock_feature_extractor,
            dataloader=tiny_dataloader,
            config=distillation_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        params_changed = False
        for name, param in trained.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed, "Teacher parameters should change after training"

    def test_checkpoint_saved(
        self,
        small_teacher: TeacherNetwork,
        tiny_dataloader: DataLoader,
        distillation_config: DistillationConfig,
        mock_feature_extractor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A checkpoint file is saved with required fields."""
        train_distillation(
            teacher=small_teacher,
            feature_extractor=mock_feature_extractor,
            dataloader=tiny_dataloader,
            config=distillation_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint_path = tmp_path / "teacher_distilled.pt"
        assert checkpoint_path.exists(), "Checkpoint file should be saved"

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "model_config" in checkpoint
        assert "config" in checkpoint
        assert "channel_indices" in checkpoint
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 1

    def test_checkpoint_loadable(
        self,
        small_teacher: TeacherNetwork,
        tiny_dataloader: DataLoader,
        distillation_config: DistillationConfig,
        mock_feature_extractor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Saved checkpoint can be loaded into a fresh teacher."""
        train_distillation(
            teacher=small_teacher,
            feature_extractor=mock_feature_extractor,
            dataloader=tiny_dataloader,
            config=distillation_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        fresh_teacher = TeacherNetwork(
            internal_channels=16,
            output_channels=32,
            num_levels=2,
            block_depths=(1, 1),
            kernel_sizes=(3, 3),
        )
        checkpoint = torch.load(tmp_path / "teacher_distilled.pt", weights_only=False)
        fresh_teacher.load_state_dict(checkpoint["model_state_dict"])

        x = torch.randn(1, 3, 256, 256)
        small_teacher.eval()
        fresh_teacher.eval()
        for a, b in zip(small_teacher(x), fresh_teacher(x), strict=True):
            assert torch.equal(a, b)

    def test_returns_teacher(
        self,
        small_teacher: TeacherNetwork,
        tiny_dataloader: DataLoader,
        distillation_config: DistillationConfig,
        mock_feature_extractor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """train_distillation returns the (trained) teacher network."""
        result = train_distillation(
            teacher=small_teacher,
            feature_extractor=mock_feature_extractor,
            dataloader=tiny_dataloader,
            config=distillation_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        assert result is small_teacher

    def test_loss_decreases(
        self,
        tmp_path: Path,
    ) -> None:
        """With 2 epochs, loss at epoch 2 should be <= loss at epoch 1.

        Uses a deterministic setup with fixed seed to make this reliable.
        """
        torch.manual_seed(42)

        teacher = TeacherNetwork(
            internal_channels=16,
            output_channels=32,
            num_levels=2,
            block_depths=(1, 1),
            kernel_sizes=(3, 3),
        )

        # Fixed target for deterministic loss comparison
        fixed_target = torch.randn(2, 32, 64, 64)

        mock_fe = MagicMock(spec=FeatureExtractor)
        mock_fe.training = False
        mock_fe.channel_indices = torch.arange(32)

        def mock_forward(images: torch.Tensor) -> torch.Tensor:
            return fixed_target[: images.shape[0]]

        mock_fe.side_effect = mock_forward
        mock_fe.to = MagicMock(return_value=mock_fe)

        images = torch.randn(2, 3, 256, 256)
        loader = DataLoader(DictDataset(images), batch_size=2, shuffle=False)

        config = DistillationConfig(
            backbone="wide_resnet50_2",
            num_epochs=2,
            learning_rate=1e-3,
            batch_size=2,
        )

        # Capture loss values from logging
        losses: list[float] = []
        distill_logger = logging.getLogger("mudenet.training.distillation")
        original_level = distill_logger.level

        class _LossCapture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                msg = record.getMessage()
                if "avg loss" in msg:
                    loss_str = msg.split("avg loss: ")[1]
                    losses.append(float(loss_str))

        handler = _LossCapture()
        distill_logger.addHandler(handler)
        distill_logger.setLevel(logging.INFO)

        try:
            train_distillation(
                teacher=teacher,
                feature_extractor=mock_fe,
                dataloader=loader,
                config=config,
                device="cpu",
                output_dir=str(tmp_path),
            )
        finally:
            distill_logger.removeHandler(handler)
            distill_logger.setLevel(original_level)

        assert len(losses) == 2, f"Expected 2 loss values, got {len(losses)}"
        assert losses[1] <= losses[0], (
            f"Loss should decrease: epoch 1={losses[0]:.6f}, epoch 2={losses[1]:.6f}"
        )
