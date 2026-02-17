"""Tests for train/eval mode behavior with BatchNorm models.

BatchNorm behaves differently in train vs eval mode:
- Train: uses batch statistics, updates running_mean/running_var
- Eval: uses frozen running_mean/running_var

These tests verify that:
1. ResidualBlock and TeacherNetwork produce different outputs in train vs eval
2. FeatureExtractor cannot be set to train mode
"""

from __future__ import annotations

import pytest
import torch

from mudenet.models.common import ResidualBlock
from mudenet.models.feature_extractor import FeatureExtractor
from mudenet.models.teacher import TeacherNetwork


class TestResidualBlockTrainEval:
    """Verify ResidualBlock behaves differently in train vs eval mode."""

    def test_different_output_train_vs_eval(self) -> None:
        """ResidualBlock produces different outputs in train vs eval mode.

        In train mode, BatchNorm uses batch statistics.
        In eval mode, BatchNorm uses running statistics (initialized to 0/1).
        After a forward pass in train mode, running stats are updated,
        so eval mode output should differ from train mode output.
        """
        torch.manual_seed(42)
        block = ResidualBlock(channels=32, kernel_size=3)
        x = torch.randn(4, 32, 16, 16)

        # First forward pass in train mode (updates running stats)
        block.train()
        train_out = block(x)

        # Same input in eval mode (uses running stats)
        block.eval()
        with torch.inference_mode():
            eval_out = block(x)

        assert not torch.equal(
            train_out, eval_out
        ), "Train and eval outputs should differ due to BatchNorm"

    def test_eval_mode_deterministic(self) -> None:
        """In eval mode, repeated forward passes give identical results."""
        torch.manual_seed(42)
        block = ResidualBlock(channels=32, kernel_size=3)
        x = torch.randn(4, 32, 16, 16)

        # Run in train mode first to populate running stats
        block.train()
        block(x)

        # Eval mode should be deterministic
        block.eval()
        with torch.inference_mode():
            out1 = block(x)
            out2 = block(x)

        assert torch.equal(out1, out2), "Eval mode should be deterministic"

    def test_train_mode_updates_running_stats(self) -> None:
        """Train mode updates BatchNorm running_mean and running_var."""
        torch.manual_seed(42)
        block = ResidualBlock(channels=32, kernel_size=3)

        # Record initial running stats
        initial_mean = block.bn1.running_mean.clone()

        # Forward pass in train mode
        block.train()
        x = torch.randn(4, 32, 16, 16)
        block(x)

        # Running stats should have been updated
        assert not torch.equal(
            block.bn1.running_mean, initial_mean
        ), "Train mode should update running_mean"


class TestTeacherNetworkTrainEval:
    """Verify TeacherNetwork behaves differently in train vs eval mode."""

    def test_different_output_train_vs_eval(self) -> None:
        """TeacherNetwork produces different outputs in train vs eval."""
        torch.manual_seed(42)
        model = TeacherNetwork(internal_channels=16, output_channels=32)
        x = torch.randn(2, 3, 256, 256)

        # Forward in train mode
        model.train()
        train_maps = model(x)

        # Forward in eval mode
        model.eval()
        with torch.inference_mode():
            eval_maps = model(x)

        any_different = any(
            not torch.equal(t, e)
            for t, e in zip(train_maps, eval_maps, strict=True)
        )
        assert any_different, "Train and eval outputs should differ due to BatchNorm"


class TestFeatureExtractorMode:
    """Verify FeatureExtractor always stays in eval mode."""

    def test_starts_in_eval_mode(self) -> None:
        """FeatureExtractor is in eval mode after construction."""
        fe = FeatureExtractor()
        assert not fe.training, "FeatureExtractor should start in eval mode"

    def test_train_mode_raises(self) -> None:
        """Setting FeatureExtractor to train mode raises RuntimeError."""
        fe = FeatureExtractor()
        with pytest.raises(RuntimeError, match="always frozen"):
            fe.train(True)

    def test_eval_mode_noop(self) -> None:
        """Setting FeatureExtractor to eval mode (train(False)) is a no-op."""
        fe = FeatureExtractor()
        result = fe.train(False)
        assert result is fe
        assert not fe.training

    def test_all_submodules_in_eval(self) -> None:
        """All sub-modules of FeatureExtractor are in eval mode."""
        fe = FeatureExtractor()
        for name, module in fe.named_modules():
            assert not module.training, f"Sub-module {name} is in train mode"
