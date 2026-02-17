"""Tests for the TeacherNetwork (also used as S1 and S2)."""

from __future__ import annotations

import pytest
import torch

from mudenet.models.teacher import TeacherNetwork


class TestTeacherNetwork:
    """Tests for TeacherNetwork."""

    def test_output_shapes_default(self, dummy_batch: torch.Tensor) -> None:
        """Default config produces 3 maps of (B, 128, 128, 128)."""
        model = TeacherNetwork()
        maps = model(dummy_batch)

        assert len(maps) == 3
        for m in maps:
            assert m.shape == (2, 128, 128, 128)

    def test_output_shapes_custom(self) -> None:
        """Custom config with 2 levels and 64 output channels."""
        model = TeacherNetwork(
            internal_channels=32,
            output_channels=64,
            num_levels=2,
            block_depths=(1, 1),
            kernel_sizes=(3, 5),
        )
        x = torch.randn(1, 3, 256, 256)
        maps = model(x)

        assert len(maps) == 2
        for m in maps:
            assert m.shape == (1, 64, 128, 128)

    def test_gradient_flow(self) -> None:
        """Gradients flow through all levels to the input."""
        model = TeacherNetwork(internal_channels=16, output_channels=32)
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        maps = model(x)

        # Backprop from last level
        loss = maps[-1].sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_all_levels_receive_gradients(self) -> None:
        """Each level's projection receives gradients independently."""
        model = TeacherNetwork(internal_channels=16, output_channels=32)
        x = torch.randn(1, 3, 256, 256)

        for level in range(3):
            model.zero_grad()
            maps = model(x)
            loss = maps[level].sum()
            loss.backward()

            # The projection for this level should have gradients
            proj = model.projections[level]
            assert proj.weight.grad is not None
            assert proj.weight.grad.abs().sum() > 0

    def test_receptive_field_gradient_based(self) -> None:
        """Verify receptive fields of 16, 32, 64 using gradient-based measurement.

        A pixel at the center of the output is activated, and we check how many
        input pixels have non-zero gradients. The RF is the side length of the
        smallest bounding box containing all non-zero gradient pixels.

        Uses random input (not zeros) to ensure gradient flow through BatchNorm.
        """
        torch.manual_seed(0)
        model = TeacherNetwork()
        model.eval()

        expected_rfs = [16, 32, 64]

        for level, expected_rf in enumerate(expected_rfs):
            # Use random input so BN statistics are non-degenerate
            x = torch.randn(1, 3, 256, 256, requires_grad=True)
            maps = model(x)

            # Select center pixel of the output map
            h, w = maps[level].shape[2], maps[level].shape[3]
            center_h, center_w = h // 2, w // 2

            # Backprop from center pixel
            model.zero_grad()
            maps[level][0, 0, center_h, center_w].backward(retain_graph=True)

            # Measure RF from gradient map
            grad = x.grad[0].abs().sum(dim=0)  # (256, 256)
            nonzero = (grad > 0).nonzero(as_tuple=False)

            if len(nonzero) > 0:
                min_h, max_h = nonzero[:, 0].min().item(), nonzero[:, 0].max().item()
                min_w, max_w = nonzero[:, 1].min().item(), nonzero[:, 1].max().item()
                rf_h = max_h - min_h + 1
                rf_w = max_w - min_w + 1
                measured_rf = max(rf_h, rf_w)
            else:
                measured_rf = 0

            assert measured_rf == expected_rf, (
                f"Level {level}: expected RF={expected_rf}, measured RF={measured_rf}"
            )

            x.grad.zero_()

    def test_determinism(self) -> None:
        """Same seed produces identical outputs."""
        torch.manual_seed(42)
        model1 = TeacherNetwork(internal_channels=16, output_channels=32)
        x1 = torch.randn(1, 3, 256, 256)
        out1 = model1(x1)

        torch.manual_seed(42)
        model2 = TeacherNetwork(internal_channels=16, output_channels=32)
        x2 = torch.randn(1, 3, 256, 256)
        out2 = model2(x2)

        for a, b in zip(out1, out2, strict=True):
            assert torch.equal(a, b)

    def test_parameter_count_default(self) -> None:
        """Default config has 666,496 parameters (verified in ADR-0001)."""
        model = TeacherNetwork()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == 666_496, f"Expected 666,496 params, got {param_count}"

    def test_invalid_block_depths_length(self) -> None:
        """Mismatched block_depths length raises ValueError."""
        with pytest.raises(ValueError, match="block_depths length"):
            TeacherNetwork(num_levels=3, block_depths=(1, 2))

    def test_invalid_kernel_sizes_length(self) -> None:
        """Mismatched kernel_sizes length raises ValueError."""
        with pytest.raises(ValueError, match="kernel_sizes length"):
            TeacherNetwork(num_levels=3, kernel_sizes=(3, 3))
