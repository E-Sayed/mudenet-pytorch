"""Tests for the inference pipeline (Eqs. 9-12, normalization, fusion)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from mudenet.config.schema import ModelConfig
from mudenet.inference.pipeline import (
    NormalizationStats,
    _min_max_normalize,
    compute_image_score,
    compute_normalization_stats,
    gaussian_smooth,
    score_batch,
)
from mudenet.models.autoencoder import Autoencoder
from mudenet.models.teacher import TeacherNetwork
from tests.conftest import DictDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def four_models(
    small_model_config: ModelConfig,
) -> tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork]:
    """Create all 4 networks (T, S1, A, S2) with small config."""
    torch.manual_seed(0)
    teacher = TeacherNetwork(
        internal_channels=small_model_config.internal_channels,
        output_channels=small_model_config.num_channels,
        num_levels=small_model_config.num_levels,
        block_depths=small_model_config.block_depths,
        kernel_sizes=small_model_config.kernel_sizes,
    )
    student1 = TeacherNetwork(
        internal_channels=small_model_config.internal_channels,
        output_channels=small_model_config.num_channels,
        num_levels=small_model_config.num_levels,
        block_depths=small_model_config.block_depths,
        kernel_sizes=small_model_config.kernel_sizes,
    )
    autoencoder = Autoencoder(
        output_channels=small_model_config.num_channels,
        latent_dim=small_model_config.latent_dim,
        num_levels=small_model_config.num_levels,
    )
    student2 = TeacherNetwork(
        internal_channels=small_model_config.internal_channels,
        output_channels=small_model_config.num_channels,
        num_levels=small_model_config.num_levels,
        block_depths=small_model_config.block_depths,
        kernel_sizes=small_model_config.kernel_sizes,
    )

    teacher.eval()
    teacher.requires_grad_(False)
    student1.eval()
    autoencoder.eval()
    student2.eval()

    return teacher, student1, autoencoder, student2


@pytest.fixture
def small_dataloader() -> DataLoader:
    """Tiny dataloader with 4 synthetic images."""
    torch.manual_seed(42)
    images = torch.randn(4, 3, 256, 256)
    dataset = DictDataset(images)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def sample_norm_stats() -> NormalizationStats:
    """Pre-made normalization stats for 2 levels."""
    return NormalizationStats(
        structural_min=[0.0, 0.5],
        structural_max=[10.0, 20.0],
        logical_min=[0.1, 0.2],
        logical_max=[5.0, 8.0],
    )


# ---------------------------------------------------------------------------
# _min_max_normalize
# ---------------------------------------------------------------------------


class TestMinMaxNormalize:
    """Tests for the _min_max_normalize helper."""

    def test_basic_normalization(self) -> None:
        """Values at min→0, at max→~1."""
        score = torch.tensor([0.0, 5.0, 10.0])
        result = _min_max_normalize(score, min_val=0.0, max_val=10.0)
        assert torch.allclose(result, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)

    def test_same_min_max_no_nan(self) -> None:
        """When min == max, result is 0 / eps, not NaN."""
        score = torch.tensor([5.0, 5.0])
        result = _min_max_normalize(score, min_val=5.0, max_val=5.0)
        assert not result.isnan().any()

    def test_preserves_shape(self) -> None:
        """Output shape matches input shape."""
        score = torch.randn(2, 8, 8)
        result = _min_max_normalize(score, min_val=0.0, max_val=1.0)
        assert result.shape == (2, 8, 8)


# ---------------------------------------------------------------------------
# NormalizationStats
# ---------------------------------------------------------------------------


class TestNormalizationStats:
    """Tests for NormalizationStats dataclass."""

    def test_round_trip_serialization(self, sample_norm_stats: NormalizationStats) -> None:
        """to_dict → from_dict produces identical stats."""
        d = sample_norm_stats.to_dict()
        restored = NormalizationStats.from_dict(d)
        assert restored.structural_min == sample_norm_stats.structural_min
        assert restored.structural_max == sample_norm_stats.structural_max
        assert restored.logical_min == sample_norm_stats.logical_min
        assert restored.logical_max == sample_norm_stats.logical_max

    def test_to_dict_keys(self, sample_norm_stats: NormalizationStats) -> None:
        """to_dict contains exactly the 4 expected keys."""
        d = sample_norm_stats.to_dict()
        expected_keys = {"structural_min", "structural_max", "logical_min", "logical_max"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_are_lists(self, sample_norm_stats: NormalizationStats) -> None:
        """All values in to_dict are lists of floats."""
        d = sample_norm_stats.to_dict()
        for v in d.values():
            assert isinstance(v, list)
            for item in v:
                assert isinstance(item, float)


# ---------------------------------------------------------------------------
# compute_normalization_stats
# ---------------------------------------------------------------------------


class TestComputeNormalizationStats:
    """Tests for compute_normalization_stats."""

    def test_returns_normalization_stats(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
        small_dataloader: DataLoader,
    ) -> None:
        """Returns a NormalizationStats instance with correct structure."""
        teacher, s1, ae, s2 = four_models
        stats = compute_normalization_stats(teacher, s1, ae, s2, small_dataloader, device="cpu")
        assert isinstance(stats, NormalizationStats)

    def test_correct_number_of_levels(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
        small_dataloader: DataLoader,
        small_model_config: ModelConfig,
    ) -> None:
        """Stats have entries for each level."""
        teacher, s1, ae, s2 = four_models
        stats = compute_normalization_stats(teacher, s1, ae, s2, small_dataloader, device="cpu")
        num_levels = small_model_config.num_levels
        assert len(stats.structural_min) == num_levels
        assert len(stats.structural_max) == num_levels
        assert len(stats.logical_min) == num_levels
        assert len(stats.logical_max) == num_levels

    def test_min_le_max(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
        small_dataloader: DataLoader,
    ) -> None:
        """structural_min <= structural_max and logical_min <= logical_max for each level."""
        teacher, s1, ae, s2 = four_models
        stats = compute_normalization_stats(teacher, s1, ae, s2, small_dataloader, device="cpu")
        for level in range(len(stats.structural_min)):
            assert stats.structural_min[level] <= stats.structural_max[level]
            assert stats.logical_min[level] <= stats.logical_max[level]

    def test_non_negative_scores(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
        small_dataloader: DataLoader,
    ) -> None:
        """Min values are non-negative (squared differences can't be negative)."""
        teacher, s1, ae, s2 = four_models
        stats = compute_normalization_stats(teacher, s1, ae, s2, small_dataloader, device="cpu")
        for level in range(len(stats.structural_min)):
            assert stats.structural_min[level] >= 0.0
            assert stats.logical_min[level] >= 0.0

    def test_empty_dataloader_raises(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
    ) -> None:
        """Empty dataloader raises ValueError."""
        teacher, s1, ae, s2 = four_models
        empty_loader = DataLoader(DictDataset(torch.zeros(0, 3, 256, 256)), batch_size=1)
        with pytest.raises(ValueError, match="zero batches"):
            compute_normalization_stats(teacher, s1, ae, s2, empty_loader, device="cpu")


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------


class TestScoreBatch:
    """Tests for score_batch (Eqs. 9-12 full pipeline)."""

    def test_output_shape(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
        sample_norm_stats: NormalizationStats,
    ) -> None:
        """Returns (B, H, W) anomaly map."""
        teacher, s1, ae, s2 = four_models
        torch.manual_seed(0)
        images = torch.randn(2, 3, 256, 256)
        anomaly_map = score_batch(teacher, s1, ae, s2, images, sample_norm_stats)
        assert anomaly_map.shape[0] == 2  # B
        assert anomaly_map.dim() == 3     # (B, H, W)

    def test_spatial_resolution(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
        sample_norm_stats: NormalizationStats,
    ) -> None:
        """Anomaly map spatial dims match the model's embedding map dims (128x128)."""
        teacher, s1, ae, s2 = four_models
        torch.manual_seed(0)
        images = torch.randn(1, 3, 256, 256)
        anomaly_map = score_batch(teacher, s1, ae, s2, images, sample_norm_stats)
        assert anomaly_map.shape == (1, 128, 128)

    def test_identical_models_produce_zero(
        self,
        small_model_config: ModelConfig,
    ) -> None:
        """When T==S1 and A==S2, both branches produce zero scores.

        Uses the same TeacherNetwork instance for T and S1 (structural=0)
        and another same instance for A and S2 (logical=0). The autoencoder
        param accepts a TeacherNetwork via duck typing since both have
        forward(x) → list[Tensor].
        """
        torch.manual_seed(99)
        teacher = TeacherNetwork(
            internal_channels=small_model_config.internal_channels,
            output_channels=small_model_config.num_channels,
            num_levels=small_model_config.num_levels,
            block_depths=small_model_config.block_depths,
            kernel_sizes=small_model_config.kernel_sizes,
        )
        # Second TeacherNetwork used as both "autoencoder" and S2
        student2 = TeacherNetwork(
            internal_channels=small_model_config.internal_channels,
            output_channels=small_model_config.num_channels,
            num_levels=small_model_config.num_levels,
            block_depths=small_model_config.block_depths,
            kernel_sizes=small_model_config.kernel_sizes,
        )
        teacher.eval()
        student2.eval()

        norm_stats = NormalizationStats(
            structural_min=[0.0] * small_model_config.num_levels,
            structural_max=[1.0] * small_model_config.num_levels,
            logical_min=[0.0] * small_model_config.num_levels,
            logical_max=[1.0] * small_model_config.num_levels,
        )

        images = torch.randn(1, 3, 256, 256)
        # T==S1 → structural score = 0; "A"==S2 → logical score = 0
        anomaly_map = score_batch(
            teacher, teacher, student2, student2, images, norm_stats  # type: ignore[arg-type]
        )

        # Both branches zero raw → (0 - 0) / (1 - 0 + eps) ≈ 0
        assert torch.allclose(anomaly_map, torch.zeros_like(anomaly_map), atol=1e-6)

    def test_with_computed_norm_stats(
        self,
        four_models: tuple[TeacherNetwork, TeacherNetwork, Autoencoder, TeacherNetwork],
        small_dataloader: DataLoader,
    ) -> None:
        """Full pipeline: compute stats then score — no errors."""
        teacher, s1, ae, s2 = four_models
        stats = compute_normalization_stats(teacher, s1, ae, s2, small_dataloader, device="cpu")

        torch.manual_seed(0)
        images = torch.randn(2, 3, 256, 256)
        anomaly_map = score_batch(teacher, s1, ae, s2, images, stats)
        assert anomaly_map.shape[0] == 2
        assert anomaly_map.dim() == 3


# ---------------------------------------------------------------------------
# compute_image_score
# ---------------------------------------------------------------------------


class TestComputeImageScore:
    """Tests for compute_image_score."""

    def test_output_shape(self) -> None:
        """Returns (B,) tensor."""
        anomaly_map = torch.randn(4, 8, 8)
        result = compute_image_score(anomaly_map)
        assert result.shape == (4,)

    def test_single_image(self) -> None:
        """Works with B=1."""
        anomaly_map = torch.randn(1, 16, 16)
        result = compute_image_score(anomaly_map)
        assert result.shape == (1,)

    def test_returns_max_of_anomaly_map(self) -> None:
        """Image score is the max pixel value per image.

        Set a known max in a specific pixel and verify.
        """
        anomaly_map = torch.zeros(2, 4, 4)
        anomaly_map[0, 1, 2] = 7.5
        anomaly_map[1, 3, 0] = 3.2
        result = compute_image_score(anomaly_map)
        assert result[0].item() == pytest.approx(7.5)
        assert result[1].item() == pytest.approx(3.2)

    def test_all_zeros(self) -> None:
        """All-zero anomaly map produces zero image score."""
        anomaly_map = torch.zeros(3, 8, 8)
        result = compute_image_score(anomaly_map)
        assert (result == 0.0).all()

    def test_negative_values(self) -> None:
        """Handles negative values correctly (max over negatives)."""
        anomaly_map = torch.full((1, 4, 4), -1.0)
        anomaly_map[0, 2, 2] = -0.5
        result = compute_image_score(anomaly_map)
        assert result[0].item() == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# gaussian_smooth
# ---------------------------------------------------------------------------


class TestGaussianSmooth:
    """Tests for gaussian_smooth."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        anomaly_map = torch.randn(2, 32, 32)
        result = gaussian_smooth(anomaly_map, sigma=4.0)
        assert result.shape == anomaly_map.shape

    def test_preserves_constant_map(self) -> None:
        """Smoothing a constant map returns the same constant."""
        anomaly_map = torch.full((1, 64, 64), 5.0)
        result = gaussian_smooth(anomaly_map, sigma=4.0)
        assert torch.allclose(result, anomaly_map, atol=1e-5)

    def test_reduces_spike(self) -> None:
        """A single-pixel spike is reduced by smoothing."""
        anomaly_map = torch.zeros(1, 64, 64)
        anomaly_map[0, 32, 32] = 100.0
        result = gaussian_smooth(anomaly_map, sigma=4.0)
        # Peak should be reduced
        assert result[0, 32, 32].item() < 100.0
        # Energy should spread to neighbors
        assert result[0, 31, 32].item() > 0.0

    def test_non_negative_preserved(self) -> None:
        """Non-negative input produces non-negative output."""
        torch.manual_seed(0)
        anomaly_map = torch.rand(2, 32, 32)  # [0, 1)
        result = gaussian_smooth(anomaly_map, sigma=2.0)
        assert (result >= -1e-6).all()

    def test_zero_sigma_raises(self) -> None:
        """sigma=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            gaussian_smooth(torch.randn(1, 8, 8), sigma=0.0)

    def test_negative_sigma_raises(self) -> None:
        """Negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            gaussian_smooth(torch.randn(1, 8, 8), sigma=-1.0)

    def test_small_sigma_minimal_effect(self) -> None:
        """Very small sigma has minimal smoothing effect."""
        torch.manual_seed(42)
        anomaly_map = torch.randn(1, 32, 32)
        result = gaussian_smooth(anomaly_map, sigma=0.1)
        assert torch.allclose(result, anomaly_map, atol=1e-3)

    def test_large_map(self) -> None:
        """Works on 256x256 maps (production size)."""
        anomaly_map = torch.randn(4, 256, 256)
        result = gaussian_smooth(anomaly_map, sigma=4.0)
        assert result.shape == (4, 256, 256)
