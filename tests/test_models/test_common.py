"""Tests for shared building blocks (Stem and ResidualBlock)."""

from __future__ import annotations

import pytest
import torch

from mudenet.models.common import ResidualBlock, Stem


class TestStem:
    """Tests for the Stem module."""

    def test_output_shape_default(self) -> None:
        """Stem reduces 256x256 -> 128x128 at 64 channels."""
        stem = Stem(out_channels=64)
        x = torch.randn(2, 3, 256, 256)
        out = stem(x)
        assert out.shape == (2, 64, 128, 128)

    def test_output_shape_custom_channels(self) -> None:
        """Stem works with non-default channel count."""
        stem = Stem(out_channels=32)
        x = torch.randn(1, 3, 256, 256)
        out = stem(x)
        assert out.shape == (1, 32, 128, 128)

    def test_spatial_halving(self) -> None:
        """Stem halves spatial dimensions via AvgPool."""
        stem = Stem(out_channels=16)
        x = torch.randn(1, 3, 128, 128)
        out = stem(x)
        assert out.shape == (1, 16, 64, 64)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the stem."""
        stem = Stem(out_channels=64)
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        out = stem(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestResidualBlock:
    """Tests for the ResidualBlock module."""

    def test_output_shape_k3(self) -> None:
        """ResidualBlock with k=3 preserves spatial dimensions."""
        block = ResidualBlock(channels=64, kernel_size=3)
        x = torch.randn(2, 64, 128, 128)
        out = block(x)
        assert out.shape == (2, 64, 128, 128)

    def test_output_shape_k5(self) -> None:
        """ResidualBlock with k=5 preserves spatial dimensions."""
        block = ResidualBlock(channels=64, kernel_size=5)
        x = torch.randn(2, 64, 128, 128)
        out = block(x)
        assert out.shape == (2, 64, 128, 128)

    def test_residual_connection(self) -> None:
        """ResidualBlock output differs from identity (non-trivial transform)."""
        block = ResidualBlock(channels=16, kernel_size=3)
        x = torch.randn(1, 16, 32, 32)
        out = block(x)
        # Output should not be identical to input (residual adds learned features)
        assert not torch.equal(x, out)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the residual block."""
        block = ResidualBlock(channels=32, kernel_size=3)
        x = torch.randn(1, 32, 64, 64, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_various_kernel_sizes(self, kernel_size: int) -> None:
        """ResidualBlock preserves spatial dims for various kernel sizes."""
        block = ResidualBlock(channels=16, kernel_size=kernel_size)
        x = torch.randn(1, 16, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_even_kernel_size_raises(self) -> None:
        """Even kernel_size is rejected with ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            ResidualBlock(channels=64, kernel_size=4)

    def test_stacked_blocks(self) -> None:
        """Multiple stacked ResidualBlocks preserve dimensions."""
        blocks = torch.nn.Sequential(
            ResidualBlock(32, kernel_size=3),
            ResidualBlock(32, kernel_size=3),
            ResidualBlock(32, kernel_size=5),
        )
        x = torch.randn(1, 32, 64, 64)
        out = blocks(x)
        assert out.shape == (1, 32, 64, 64)
