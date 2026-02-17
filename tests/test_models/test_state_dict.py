"""Tests for state_dict save/load round-trip on all models.

Verifies that saving and loading state_dict produces identical model outputs.
Especially important for FeatureExtractor where channel_indices is a registered
buffer that must survive serialization.
"""

from __future__ import annotations

import io

import torch

from mudenet.models.autoencoder import Autoencoder
from mudenet.models.feature_extractor import FeatureExtractor
from mudenet.models.teacher import TeacherNetwork


class TestStateDictRoundTrip:
    """Verify save/load produces identical outputs for all models."""

    def test_teacher_round_trip(self) -> None:
        """TeacherNetwork state_dict round-trip produces identical outputs."""
        torch.manual_seed(42)
        original = TeacherNetwork(internal_channels=16, output_channels=32)
        original.eval()

        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            original_out = original(x)

        # Save to buffer
        buf = io.BytesIO()
        torch.save(original.state_dict(), buf)
        buf.seek(0)

        # Load into fresh model
        loaded = TeacherNetwork(internal_channels=16, output_channels=32)
        loaded.load_state_dict(torch.load(buf, weights_only=True))
        loaded.eval()

        with torch.inference_mode():
            loaded_out = loaded(x)

        for a, b in zip(original_out, loaded_out, strict=True):
            assert torch.equal(a, b), "TeacherNetwork outputs differ after round-trip"

    def test_autoencoder_round_trip(self) -> None:
        """Autoencoder state_dict round-trip produces identical outputs."""
        torch.manual_seed(42)
        original = Autoencoder(output_channels=32, latent_dim=8, num_levels=2)
        original.eval()

        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            original_out = original(x)

        # Save to buffer
        buf = io.BytesIO()
        torch.save(original.state_dict(), buf)
        buf.seek(0)

        # Load into fresh model
        loaded = Autoencoder(output_channels=32, latent_dim=8, num_levels=2)
        loaded.load_state_dict(torch.load(buf, weights_only=True))
        loaded.eval()

        with torch.inference_mode():
            loaded_out = loaded(x)

        for a, b in zip(original_out, loaded_out, strict=True):
            assert torch.equal(a, b), "Autoencoder outputs differ after round-trip"

    def test_feature_extractor_round_trip(self) -> None:
        """FeatureExtractor state_dict round-trip preserves channel_indices buffer."""
        original = FeatureExtractor(output_channels=128, seed=42)

        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            original_out = original(x)

        # Save to buffer
        buf = io.BytesIO()
        torch.save(original.state_dict(), buf)
        buf.seek(0)

        # Load into fresh model (different seed to prove buffer is restored)
        loaded = FeatureExtractor(output_channels=128, seed=999)
        loaded.load_state_dict(torch.load(buf, weights_only=True))

        # Verify buffer was restored
        assert torch.equal(
            original.channel_indices, loaded.channel_indices
        ), "channel_indices buffer not preserved after round-trip"

        with torch.inference_mode():
            loaded_out = loaded(x)

        assert torch.equal(
            original_out, loaded_out
        ), "FeatureExtractor outputs differ after round-trip"

    def test_feature_extractor_channel_indices_in_state_dict(self) -> None:
        """channel_indices appears in the state_dict as a buffer."""
        fe = FeatureExtractor(output_channels=128, seed=42)
        state = fe.state_dict()
        assert "channel_indices" in state, "channel_indices missing from state_dict"
        assert state["channel_indices"].shape == (128,)
