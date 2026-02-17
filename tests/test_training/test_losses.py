"""Tests for loss functions (Eq. 3, 5, 7, 8, 16)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from mudenet.training.losses import (
    _pairwise_loss,
    autoencoder_loss,
    composite_loss,
    distillation_loss,
    logical_loss,
    structural_loss,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identical_maps() -> list[Tensor]:
    """Two-level maps with known values (all ones)."""
    return [torch.ones(2, 4, 8, 8) for _ in range(2)]


@pytest.fixture
def different_maps_a() -> list[Tensor]:
    """Two-level maps A (all zeros)."""
    return [torch.zeros(2, 4, 8, 8) for _ in range(2)]


@pytest.fixture
def different_maps_b() -> list[Tensor]:
    """Two-level maps B (all ones)."""
    return [torch.ones(2, 4, 8, 8) for _ in range(2)]


# ---------------------------------------------------------------------------
# _pairwise_loss
# ---------------------------------------------------------------------------


class TestPairwiseLoss:
    """Tests for the shared _pairwise_loss helper."""

    def test_zero_loss_identical(self, identical_maps: list[Tensor]) -> None:
        """Identical maps produce zero loss."""
        loss = _pairwise_loss(identical_maps, identical_maps)
        assert loss.item() == 0.0

    def test_positive_loss_different(
        self, different_maps_a: list[Tensor], different_maps_b: list[Tensor]
    ) -> None:
        """Different maps produce positive loss."""
        loss = _pairwise_loss(different_maps_a, different_maps_b)
        assert loss.item() > 0.0

    def test_scalar_output(self, identical_maps: list[Tensor]) -> None:
        """Output is a scalar tensor (0-dim)."""
        loss = _pairwise_loss(identical_maps, identical_maps)
        assert loss.dim() == 0

    def test_known_value(self) -> None:
        """Hand-computed example.

        maps_a: all zeros, shape (1, 2, 2, 2)
        maps_b: all ones, shape (1, 2, 2, 2)
        Per-level: (0 - 1)^2 = 1 per element, sum over C=2 -> 2 per spatial pos,
                   mean over B=1, H=2, W=2 -> 2.0
        Average over L=1 level -> 2.0
        """
        a = [torch.zeros(1, 2, 2, 2)]
        b = [torch.ones(1, 2, 2, 2)]
        loss = _pairwise_loss(a, b)
        assert torch.isclose(loss, torch.tensor(2.0))

    def test_known_value_multi_level(self) -> None:
        """Hand-computed example with 2 levels.

        Level 0: a=0, b=1 -> per-level loss = C = 4 (since C=4 channels)
        Level 1: a=0, b=2 -> (0-2)^2=4 per element, sum over C=4 -> 16,
                 mean over B=2, H=8, W=8 -> 16.0
        Average over L=2 levels -> (4.0 + 16.0) / 2 = 10.0
        """
        a = [torch.zeros(2, 4, 8, 8), torch.zeros(2, 4, 8, 8)]
        b = [torch.ones(2, 4, 8, 8), 2 * torch.ones(2, 4, 8, 8)]
        loss = _pairwise_loss(a, b)
        assert torch.isclose(loss, torch.tensor(10.0))

    def test_symmetry(self) -> None:
        """Loss is symmetric: L(a, b) == L(b, a)."""
        torch.manual_seed(0)
        a = [torch.randn(2, 4, 8, 8)]
        b = [torch.randn(2, 4, 8, 8)]
        assert torch.isclose(_pairwise_loss(a, b), _pairwise_loss(b, a))

    def test_mismatched_levels_raises(self) -> None:
        """Mismatched number of levels raises ValueError."""
        a = [torch.zeros(1, 2, 2, 2)]
        b = [torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2)]
        with pytest.raises(ValueError, match="Number of levels must match"):
            _pairwise_loss(a, b)

    def test_empty_levels_raises(self) -> None:
        """Empty maps list raises ValueError."""
        with pytest.raises(ValueError, match="At least one level"):
            _pairwise_loss([], [])

    def test_gradient_flow(self) -> None:
        """Gradients flow through the loss."""
        a = [torch.randn(1, 4, 8, 8, requires_grad=True)]
        b = [torch.randn(1, 4, 8, 8)]
        loss = _pairwise_loss(a, b)
        loss.backward()
        assert a[0].grad is not None
        assert a[0].grad.shape == a[0].shape


# ---------------------------------------------------------------------------
# distillation_loss
# ---------------------------------------------------------------------------


class TestDistillationLoss:
    """Tests for distillation_loss (Eq. 16)."""

    def test_zero_loss_identical(self) -> None:
        """Identical target and teacher maps produce zero loss."""
        target = torch.ones(2, 4, 8, 8)
        teacher_maps = [torch.ones(2, 4, 8, 8) for _ in range(3)]
        loss = distillation_loss(target, teacher_maps)
        assert loss.item() == 0.0

    def test_positive_loss_different(self) -> None:
        """Different target and teacher maps produce positive loss."""
        target = torch.zeros(2, 4, 8, 8)
        teacher_maps = [torch.ones(2, 4, 8, 8) for _ in range(3)]
        loss = distillation_loss(target, teacher_maps)
        assert loss.item() > 0.0

    def test_scalar_output(self) -> None:
        """Output is a scalar tensor."""
        target = torch.randn(1, 4, 8, 8)
        teacher_maps = [torch.randn(1, 4, 8, 8)]
        loss = distillation_loss(target, teacher_maps)
        assert loss.dim() == 0

    def test_known_value(self) -> None:
        """Hand-computed distillation loss.

        target: all zeros (1, 2, 2, 2)
        teacher_maps: [all ones, all 2s] — 2 levels
        Level 0: (0-1)^2=1 per elem, sum C=2 -> 2, mean over B,H,W -> 2.0
        Level 1: (0-2)^2=4 per elem, sum C=2 -> 8, mean over B,H,W -> 8.0
        Average: (2 + 8) / 2 = 5.0
        """
        target = torch.zeros(1, 2, 2, 2)
        teacher_maps = [torch.ones(1, 2, 2, 2), 2.0 * torch.ones(1, 2, 2, 2)]
        loss = distillation_loss(target, teacher_maps)
        assert torch.isclose(loss, torch.tensor(5.0))

    def test_same_target_all_levels(self) -> None:
        """All levels compare against the same target (single tensor)."""
        target = torch.randn(2, 4, 8, 8)
        # Make each level identical to target — should be zero
        teacher_maps = [target.clone() for _ in range(3)]
        loss = distillation_loss(target, teacher_maps)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_empty_maps_raises(self) -> None:
        """Empty teacher_maps raises ValueError."""
        target = torch.randn(1, 2, 2, 2)
        with pytest.raises(ValueError, match="At least one teacher map"):
            distillation_loss(target, [])

    def test_gradient_flow(self) -> None:
        """Gradients flow to teacher maps but not target (detached in practice)."""
        target = torch.randn(1, 4, 8, 8)
        teacher_map = torch.randn(1, 4, 8, 8, requires_grad=True)
        loss = distillation_loss(target, [teacher_map])
        loss.backward()
        assert teacher_map.grad is not None


# ---------------------------------------------------------------------------
# structural_loss, autoencoder_loss, logical_loss — thin wrappers
# ---------------------------------------------------------------------------


class TestStructuralLoss:
    """Tests for structural_loss (Eq. 3)."""

    def test_zero_loss(self, identical_maps: list[Tensor]) -> None:
        """Identical maps yield zero loss."""
        assert structural_loss(identical_maps, identical_maps).item() == 0.0

    def test_positive_loss(
        self, different_maps_a: list[Tensor], different_maps_b: list[Tensor]
    ) -> None:
        """Different maps yield positive loss."""
        assert structural_loss(different_maps_a, different_maps_b).item() > 0.0

    def test_delegates_to_pairwise(self) -> None:
        """structural_loss produces the same result as _pairwise_loss."""
        torch.manual_seed(1)
        a = [torch.randn(2, 4, 8, 8)]
        b = [torch.randn(2, 4, 8, 8)]
        assert torch.equal(structural_loss(a, b), _pairwise_loss(a, b))


class TestAutoencoderLoss:
    """Tests for autoencoder_loss (Eq. 5)."""

    def test_zero_loss(self, identical_maps: list[Tensor]) -> None:
        """Identical maps yield zero loss."""
        assert autoencoder_loss(identical_maps, identical_maps).item() == 0.0

    def test_positive_loss(
        self, different_maps_a: list[Tensor], different_maps_b: list[Tensor]
    ) -> None:
        """Different maps yield positive loss."""
        assert autoencoder_loss(different_maps_a, different_maps_b).item() > 0.0


class TestLogicalLoss:
    """Tests for logical_loss (Eq. 7)."""

    def test_zero_loss(self, identical_maps: list[Tensor]) -> None:
        """Identical maps yield zero loss."""
        assert logical_loss(identical_maps, identical_maps).item() == 0.0

    def test_positive_loss(
        self, different_maps_a: list[Tensor], different_maps_b: list[Tensor]
    ) -> None:
        """Different maps yield positive loss."""
        assert logical_loss(different_maps_a, different_maps_b).item() > 0.0


# ---------------------------------------------------------------------------
# composite_loss
# ---------------------------------------------------------------------------


class TestCompositeLoss:
    """Tests for composite_loss (Eq. 8)."""

    def test_sum_of_components(self) -> None:
        """Composite loss equals the sum of its components."""
        s = torch.tensor(1.0)
        a = torch.tensor(2.0)
        lg = torch.tensor(3.0)
        result = composite_loss(s, a, lg)
        assert torch.isclose(result, torch.tensor(6.0))

    def test_zero_components(self) -> None:
        """All-zero components produce zero composite loss."""
        z = torch.tensor(0.0)
        assert composite_loss(z, z, z).item() == 0.0

    def test_scalar_output(self) -> None:
        """Output is a scalar."""
        result = composite_loss(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
        assert result.dim() == 0

    def test_gradient_flow(self) -> None:
        """Gradients flow through all three components."""
        s = torch.tensor(1.0, requires_grad=True)
        a = torch.tensor(2.0, requires_grad=True)
        lg = torch.tensor(3.0, requires_grad=True)
        total = composite_loss(s, a, lg)
        total.backward()
        assert s.grad is not None
        assert a.grad is not None
        assert lg.grad is not None
