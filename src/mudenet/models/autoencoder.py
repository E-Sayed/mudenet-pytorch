"""Autoencoder network for MuDeNet.

Implements the asymmetric autoencoder A from paper Figure 3 (Sec. 3.1).
One shared encoder E compresses the input image to a Z-dim latent vector.
L separate decoders G^l each reconstruct one level's embedding map.

Encoder architecture (6 strided convolutions):
    Conv2d(3, 32, 3, stride=2, pad=1)     -> 32  @ 128x128
    Conv2d(32, 64, 3, stride=2, pad=1)    -> 64  @ 64x64
    Conv2d(64, C, 3, stride=2, pad=1)     -> C   @ 32x32
    Conv2d(C, C, 3, stride=2, pad=1)      -> C   @ 16x16
    Conv2d(C, C, 3, stride=2, pad=1)      -> C   @ 8x8
    Conv2d(C, Z, 8, stride=1, pad=0)      -> Z   @ 1x1

Decoder architecture (6 transposed convolutions, one per level):
    ConvTranspose2d(Z, C, 4, stride=4, pad=0)                      -> C @ 4x4
    ConvTranspose2d(C, C, 3, stride=2, pad=1, output_padding=1)    -> C @ 8x8
    ConvTranspose2d(C, C, 3, stride=2, pad=1, output_padding=1)    -> C @ 16x16
    ConvTranspose2d(C, C, 3, stride=2, pad=1, output_padding=1)    -> C @ 32x32
    ConvTranspose2d(C, C, 3, stride=2, pad=1, output_padding=1)    -> C @ 64x64
    ConvTranspose2d(C, C, 3, stride=2, pad=1, output_padding=1)    -> C @ 128x128

ReLU activation after each convolution (Figure 3 caption).

See assumptions-register.md:
    A-007 (encoder padding), A-008 (decoder stride/padding).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """Autoencoder encoder from paper Figure 3.

    Six strided convolutions compress the input image to a Z-dim latent vector.
    ReLU activation after each convolution (stated in Figure 3 caption).

    Args:
        output_channels: Embedding channels C (matches teacher output). Default 128.
        latent_dim: Latent dimensionality Z. Default 32.
    """

    def __init__(self, output_channels: int = 128, latent_dim: int = 32) -> None:
        super().__init__()

        c = output_channels
        z = latent_dim

        self.layers = nn.Sequential(
            # 3 @ 256x256 -> 32 @ 128x128
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32 @ 128x128 -> 64 @ 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 @ 64x64 -> C @ 32x32
            nn.Conv2d(64, c, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # C @ 32x32 -> C @ 16x16
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # C @ 16x16 -> C @ 8x8
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # C @ 8x8 -> Z @ 1x1 (8x8 kernel collapses spatial dims)
            nn.Conv2d(c, z, kernel_size=8, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode input image to latent vector (Eq. 4, encoder part).

        Args:
            x: Input image tensor (B, 3, 256, 256).

        Returns:
            Latent vector y of shape (B, Z).
        """
        out = self.layers(x)  # (B, Z, 1, 1)
        return out.flatten(start_dim=1)  # (B, Z)


class Decoder(nn.Module):
    """Single decoder from paper Figure 3.

    Six transposed convolutions reconstruct one embedding level's map
    from the latent vector. ReLU after each convolution (Figure 3 caption).

    Each decoder G^l produces one output X^l_A in R^{C x 128 x 128}.

    Args:
        output_channels: Embedding channels C. Default 128.
        latent_dim: Latent dimensionality Z. Default 32.
    """

    def __init__(self, output_channels: int = 128, latent_dim: int = 32) -> None:
        super().__init__()

        c = output_channels
        z = latent_dim

        self.layers = nn.Sequential(
            # Z @ 1x1 -> C @ 4x4
            nn.ConvTranspose2d(z, c, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            # C @ 4x4 -> C @ 8x8
            nn.ConvTranspose2d(c, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # C @ 8x8 -> C @ 16x16
            nn.ConvTranspose2d(c, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # C @ 16x16 -> C @ 32x32
            nn.ConvTranspose2d(c, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # C @ 32x32 -> C @ 64x64
            nn.ConvTranspose2d(c, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # C @ 64x64 -> C @ 128x128
            nn.ConvTranspose2d(c, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Decode latent vector to embedding map (Eq. 4, decoder part).

        Args:
            latent: Latent vector (B, Z).

        Returns:
            Reconstructed embedding map X^l_A of shape (B, C, 128, 128).
        """
        # Reshape latent to spatial format: (B, Z) -> (B, Z, 1, 1)
        x = latent.unsqueeze(-1).unsqueeze(-1)  # (B, Z, 1, 1)
        return self.layers(x)  # (B, C, 128, 128)


class Autoencoder(nn.Module):
    """Full autoencoder A from paper Figure 3 (Sec. 3.1, Eq. 4).

    One shared encoder E compresses the input image to a Z-dim latent vector.
    L separate decoders G^l each reconstruct one level's embedding map
    from that shared latent representation.

    Args:
        output_channels: Embedding channels C (must match teacher output). Default 128.
        latent_dim: Latent dimensionality Z. Default 32.
        num_levels: Number of decoder levels L (one decoder per level). Default 3.
    """

    def __init__(
        self,
        output_channels: int = 128,
        latent_dim: int = 32,
        num_levels: int = 3,
    ) -> None:
        super().__init__()

        self.num_levels = num_levels
        self.encoder = Encoder(output_channels=output_channels, latent_dim=latent_dim)
        self.decoders = nn.ModuleList([
            Decoder(output_channels=output_channels, latent_dim=latent_dim)
            for _ in range(num_levels)
        ])

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass: encode image, decode to L embedding maps (Eq. 4).

        Args:
            x: Input image tensor (B, 3, 256, 256).

        Returns:
            List of L reconstructed embedding maps [X^1_A, ..., X^L_A],
            each (B, C, 128, 128).
        """
        latent = self.encoder(x)  # (B, Z)
        autoencoder_maps = [decoder(latent) for decoder in self.decoders]
        return autoencoder_maps
