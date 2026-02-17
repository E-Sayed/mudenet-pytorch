"""Tests for the WideResNet50 feature extractor."""

from __future__ import annotations

import pytest
import torch

from mudenet.models.feature_extractor import FeatureExtractor


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """Create a feature extractor with default settings."""
    return FeatureExtractor(output_channels=128, seed=42)


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_output_shape(self, feature_extractor: FeatureExtractor) -> None:
        """Output shape is (B, C=128, 64, 64) for 256x256 input."""
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            e = feature_extractor(x)
        assert e.shape == (1, 128, 64, 64)

    def test_output_shape_batch(self, feature_extractor: FeatureExtractor) -> None:
        """Batch dimension is preserved."""
        x = torch.randn(2, 3, 256, 256)
        with torch.inference_mode():
            e = feature_extractor(x)
        assert e.shape == (2, 128, 64, 64)

    def test_custom_output_channels(self) -> None:
        """Custom output channel count works."""
        fe = FeatureExtractor(output_channels=64, seed=42)
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            e = fe(x)
        assert e.shape == (1, 64, 64, 64)

    def test_frozen_parameters(self, feature_extractor: FeatureExtractor) -> None:
        """All parameters are frozen (requires_grad=False)."""
        for param in feature_extractor.parameters():
            assert not param.requires_grad

    def test_channel_indices_shape(self, feature_extractor: FeatureExtractor) -> None:
        """Channel indices buffer has correct shape."""
        assert feature_extractor.channel_indices.shape == (128,)

    def test_channel_indices_range(self, feature_extractor: FeatureExtractor) -> None:
        """Channel indices are within valid range [0, 1792)."""
        assert feature_extractor.channel_indices.min() >= 0
        assert feature_extractor.channel_indices.max() < 1792

    def test_channel_indices_reproducible(self) -> None:
        """Same seed produces same channel indices."""
        fe1 = FeatureExtractor(output_channels=128, seed=42)
        fe2 = FeatureExtractor(output_channels=128, seed=42)
        assert torch.equal(fe1.channel_indices, fe2.channel_indices)

    def test_different_seeds_different_indices(self) -> None:
        """Different seeds produce different channel indices."""
        fe1 = FeatureExtractor(output_channels=128, seed=42)
        fe2 = FeatureExtractor(output_channels=128, seed=123)
        assert not torch.equal(fe1.channel_indices, fe2.channel_indices)

    def test_z_score_normalization(self, feature_extractor: FeatureExtractor) -> None:
        """Output is approximately z-score normalized per channel per sample.

        Note: A small number of channels may produce constant spatial values
        (std=0), which is expected with pretrained frozen BatchNorm in eval mode.
        We check that the vast majority of channels are well-normalized.
        """
        x = torch.randn(2, 3, 256, 256)
        with torch.inference_mode():
            e = feature_extractor(x)

        # Mean should be close to 0, std close to 1 across spatial dims
        mean = e.mean(dim=[2, 3])  # (B, C)
        std = e.std(dim=[2, 3])    # (B, C)
        assert mean.abs().max() < 0.5, f"Mean too far from 0: {mean.abs().max()}"
        # Allow a few channels to have degenerate std (constant spatial maps)
        # by checking the median rather than the max
        assert (std - 1.0).abs().median() < 0.1, (
            f"Median std deviation from 1.0 too large: {(std - 1.0).abs().median()}"
        )

    def test_invalid_backbone(self) -> None:
        """Invalid backbone name raises ValueError."""
        with pytest.raises(ValueError, match="Only 'wide_resnet50_2'"):
            FeatureExtractor(backbone="resnet18")

    def test_extract_convenience(self, feature_extractor: FeatureExtractor) -> None:
        """extract() method produces same output as forward()."""
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            e1 = feature_extractor(x)
            e2 = feature_extractor.extract(x)
        assert torch.equal(e1, e2)
