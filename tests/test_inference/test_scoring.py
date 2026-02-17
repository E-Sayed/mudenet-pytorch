"""Tests for per-level anomaly scoring functions (Eqs. 9-10)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from mudenet.inference.scoring import (
    _pairwise_score,
    logical_score,
    structural_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identical_maps() -> list[Tensor]:
    """Two-level maps with known values (all ones)."""
    return [torch.ones(2, 4, 8, 8) for _ in range(2)]


# ---------------------------------------------------------------------------
# _pairwise_score
# ---------------------------------------------------------------------------


class TestPairwiseScore:
    """Tests for the shared _pairwise_score helper."""

    def test_output_is_list_of_tensors(self, identical_maps: list[Tensor]) -> None:
        """Returns a list of tensors."""
        result = _pairwise_score(identical_maps, identical_maps)
        assert isinstance(result, list)
        for t in result:
            assert isinstance(t, Tensor)

    def test_output_shapes(self) -> None:
        """Each output tensor has shape (B, H, W) — channel dim is removed."""
        maps = [torch.randn(3, 8, 16, 16), torch.randn(3, 8, 16, 16)]
        result = _pairwise_score(maps, maps)
        assert len(result) == 2
        for t in result:
            assert t.shape == (3, 16, 16)

    def test_identical_maps_produce_zero(self, identical_maps: list[Tensor]) -> None:
        """Identical maps produce zero scores at all pixels."""
        result = _pairwise_score(identical_maps, identical_maps)
        for t in result:
            assert (t == 0.0).all()

    def test_known_value_unit_difference(self) -> None:
        """Maps differing by 1 in all channels: score should equal C.

        maps_a: all zeros (B=1, C=4, H=2, W=2)
        maps_b: all ones (B=1, C=4, H=2, W=2)
        Per pixel: (0-1)^2 = 1, summed over C=4 channels → 4.0
        """
        a = [torch.zeros(1, 4, 2, 2)]
        b = [torch.ones(1, 4, 2, 2)]
        result = _pairwise_score(a, b)
        assert len(result) == 1
        expected = torch.full((1, 2, 2), 4.0)
        assert torch.allclose(result[0], expected)

    def test_known_value_non_unit_difference(self) -> None:
        """Maps differing by 3: score should equal C * 9.

        maps_a: all zeros (B=1, C=2, H=2, W=2)
        maps_b: all threes (B=1, C=2, H=2, W=2)
        Per pixel: (0-3)^2 = 9, summed over C=2 → 18.0
        """
        a = [torch.zeros(1, 2, 2, 2)]
        b = [3.0 * torch.ones(1, 2, 2, 2)]
        result = _pairwise_score(a, b)
        expected = torch.full((1, 2, 2), 18.0)
        assert torch.allclose(result[0], expected)

    def test_symmetry(self) -> None:
        """Score is symmetric: score(a, b) == score(b, a)."""
        torch.manual_seed(0)
        a = [torch.randn(2, 4, 8, 8)]
        b = [torch.randn(2, 4, 8, 8)]
        result_ab = _pairwise_score(a, b)
        result_ba = _pairwise_score(b, a)
        for s_ab, s_ba in zip(result_ab, result_ba, strict=True):
            assert torch.allclose(s_ab, s_ba)

    def test_non_negative(self) -> None:
        """Scores are always non-negative (squared differences)."""
        torch.manual_seed(1)
        a = [torch.randn(2, 4, 8, 8)]
        b = [torch.randn(2, 4, 8, 8)]
        result = _pairwise_score(a, b)
        for t in result:
            assert (t >= 0.0).all()

    def test_mismatched_levels_raises(self) -> None:
        """Mismatched number of levels raises ValueError."""
        a = [torch.zeros(1, 2, 2, 2)]
        b = [torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2)]
        with pytest.raises(ValueError, match="Number of levels must match"):
            _pairwise_score(a, b)

    def test_empty_levels_raises(self) -> None:
        """Empty maps list raises ValueError."""
        with pytest.raises(ValueError, match="At least one level"):
            _pairwise_score([], [])

    def test_multi_level(self) -> None:
        """Multi-level output has correct count and shapes.

        3 levels, B=2, C=8, H=4, W=4 → 3 tensors of (2, 4, 4).
        """
        a = [torch.randn(2, 8, 4, 4) for _ in range(3)]
        b = [torch.randn(2, 8, 4, 4) for _ in range(3)]
        result = _pairwise_score(a, b)
        assert len(result) == 3
        for t in result:
            assert t.shape == (2, 4, 4)


# ---------------------------------------------------------------------------
# structural_score
# ---------------------------------------------------------------------------


class TestStructuralScore:
    """Tests for structural_score (Eq. 9)."""

    def test_output_shapes(self) -> None:
        """Returns L tensors of shape (B, H, W)."""
        maps = [torch.randn(2, 16, 8, 8) for _ in range(3)]
        result = structural_score(maps, maps)
        assert len(result) == 3
        for t in result:
            assert t.shape == (2, 8, 8)

    def test_identical_maps_zero(self, identical_maps: list[Tensor]) -> None:
        """Identical teacher and student maps produce zero scores."""
        result = structural_score(identical_maps, identical_maps)
        for t in result:
            assert (t == 0.0).all()

    def test_known_value_c_channels(self) -> None:
        """Difference of 1 across C channels produces score = C.

        Teacher: all zeros (B=1, C=6, H=2, W=2)
        Student: all ones (B=1, C=6, H=2, W=2)
        Score per pixel: 6.0
        """
        teacher = [torch.zeros(1, 6, 2, 2)]
        student = [torch.ones(1, 6, 2, 2)]
        result = structural_score(teacher, student)
        expected = torch.full((1, 2, 2), 6.0)
        assert torch.allclose(result[0], expected)

    def test_delegates_to_pairwise(self) -> None:
        """structural_score produces the same result as _pairwise_score."""
        torch.manual_seed(2)
        a = [torch.randn(2, 4, 8, 8)]
        b = [torch.randn(2, 4, 8, 8)]
        expected = _pairwise_score(a, b)
        actual = structural_score(a, b)
        for e, ac in zip(expected, actual, strict=True):
            assert torch.equal(e, ac)

    def test_level_count_mismatch_raises(self) -> None:
        """Mismatched level counts raise ValueError."""
        a = [torch.randn(1, 4, 4, 4)]
        b = [torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)]
        with pytest.raises(ValueError, match="Number of levels must match"):
            structural_score(a, b)


# ---------------------------------------------------------------------------
# logical_score
# ---------------------------------------------------------------------------


class TestLogicalScore:
    """Tests for logical_score (Eq. 10)."""

    def test_output_shapes(self) -> None:
        """Returns L tensors of shape (B, H, W)."""
        maps = [torch.randn(2, 16, 8, 8) for _ in range(3)]
        result = logical_score(maps, maps)
        assert len(result) == 3
        for t in result:
            assert t.shape == (2, 8, 8)

    def test_identical_maps_zero(self, identical_maps: list[Tensor]) -> None:
        """Identical autoencoder and student maps produce zero scores."""
        result = logical_score(identical_maps, identical_maps)
        for t in result:
            assert (t == 0.0).all()

    def test_known_value_c_channels(self) -> None:
        """Difference of 1 across C channels produces score = C.

        Autoencoder: all zeros (B=1, C=10, H=3, W=3)
        Student: all ones (B=1, C=10, H=3, W=3)
        Score per pixel: 10.0
        """
        ae = [torch.zeros(1, 10, 3, 3)]
        student = [torch.ones(1, 10, 3, 3)]
        result = logical_score(ae, student)
        expected = torch.full((1, 3, 3), 10.0)
        assert torch.allclose(result[0], expected)

    def test_delegates_to_pairwise(self) -> None:
        """logical_score produces the same result as _pairwise_score."""
        torch.manual_seed(3)
        a = [torch.randn(2, 4, 8, 8)]
        b = [torch.randn(2, 4, 8, 8)]
        expected = _pairwise_score(a, b)
        actual = logical_score(a, b)
        for e, ac in zip(expected, actual, strict=True):
            assert torch.equal(e, ac)

    def test_level_count_mismatch_raises(self) -> None:
        """Mismatched level counts raise ValueError."""
        a = [torch.randn(1, 4, 4, 4)]
        b = [torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)]
        with pytest.raises(ValueError, match="Number of levels must match"):
            logical_score(a, b)
