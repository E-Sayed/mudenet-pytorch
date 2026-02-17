"""Tests for the Autoencoder (Encoder, Decoder, and full Autoencoder)."""

from __future__ import annotations

import torch

from mudenet.models.autoencoder import Autoencoder, Decoder, Encoder


class TestEncoder:
    """Tests for the Encoder module."""

    def test_output_shape_default(self) -> None:
        """Default encoder produces (B, Z=32) latent vector."""
        encoder = Encoder(output_channels=128, latent_dim=32)
        x = torch.randn(2, 3, 256, 256)
        latent = encoder(x)
        assert latent.shape == (2, 32)

    def test_output_shape_custom(self) -> None:
        """Custom latent dim produces correct shape."""
        encoder = Encoder(output_channels=64, latent_dim=16)
        x = torch.randn(1, 3, 256, 256)
        latent = encoder(x)
        assert latent.shape == (1, 16)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the encoder."""
        encoder = Encoder(output_channels=64, latent_dim=16)
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        latent = encoder(x)
        loss = latent.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestDecoder:
    """Tests for the Decoder module."""

    def test_output_shape_default(self) -> None:
        """Default decoder produces (B, C=128, 128, 128)."""
        decoder = Decoder(output_channels=128, latent_dim=32)
        latent = torch.randn(2, 32)
        out = decoder(latent)
        assert out.shape == (2, 128, 128, 128)

    def test_output_shape_custom(self) -> None:
        """Custom decoder produces correct shape."""
        decoder = Decoder(output_channels=64, latent_dim=16)
        latent = torch.randn(1, 16)
        out = decoder(latent)
        assert out.shape == (1, 64, 128, 128)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the decoder."""
        decoder = Decoder(output_channels=64, latent_dim=16)
        latent = torch.randn(1, 16, requires_grad=True)
        out = decoder(latent)
        loss = out.sum()
        loss.backward()
        assert latent.grad is not None
        assert latent.grad.shape == latent.shape


class TestAutoencoder:
    """Tests for the full Autoencoder module."""

    def test_output_shapes_default(self) -> None:
        """Default autoencoder produces 3 maps of (B, 128, 128, 128)."""
        ae = Autoencoder(output_channels=128, latent_dim=32, num_levels=3)
        x = torch.randn(2, 3, 256, 256)
        maps = ae(x)

        assert len(maps) == 3
        for m in maps:
            assert m.shape == (2, 128, 128, 128)

    def test_output_shapes_custom(self) -> None:
        """Custom autoencoder with 2 levels."""
        ae = Autoencoder(output_channels=64, latent_dim=16, num_levels=2)
        x = torch.randn(1, 3, 256, 256)
        maps = ae(x)

        assert len(maps) == 2
        for m in maps:
            assert m.shape == (1, 64, 128, 128)

    def test_shared_encoder(self) -> None:
        """All decoders share the same encoder (single latent computation)."""
        ae = Autoencoder(output_channels=64, latent_dim=16, num_levels=3)
        # Verify there is exactly one encoder
        assert hasattr(ae, "encoder")
        assert isinstance(ae.encoder, Encoder)
        # And L decoders
        assert len(ae.decoders) == 3

    def test_gradient_flow(self) -> None:
        """Gradients flow from all decoder outputs to input."""
        ae = Autoencoder(output_channels=32, latent_dim=8, num_levels=2)
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        maps = ae(x)

        loss = sum(m.sum() for m in maps)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_determinism(self) -> None:
        """Same seed produces identical outputs."""
        torch.manual_seed(42)
        ae1 = Autoencoder(output_channels=32, latent_dim=8, num_levels=2)
        x1 = torch.randn(1, 3, 256, 256)
        out1 = ae1(x1)

        torch.manual_seed(42)
        ae2 = Autoencoder(output_channels=32, latent_dim=8, num_levels=2)
        x2 = torch.randn(1, 3, 256, 256)
        out2 = ae2(x2)

        for a, b in zip(out1, out2, strict=True):
            assert torch.equal(a, b)

    def test_decoder_outputs_differ(self) -> None:
        """Different decoders produce different outputs (they're separate networks)."""
        ae = Autoencoder(output_channels=32, latent_dim=8, num_levels=3)
        x = torch.randn(1, 3, 256, 256)
        maps = ae(x)

        # At least some pairs should differ (different random weights)
        all_equal = all(torch.equal(maps[0], maps[i]) for i in range(1, len(maps)))
        assert not all_equal, "All decoder outputs are identical — decoders may be shared"
