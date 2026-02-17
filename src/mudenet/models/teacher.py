"""Teacher (and student) network for MuDeNet.

Implements the multi-scale embedding network from paper Figure 2 (Sec. 3.1).
The same architecture is used for the teacher T, structural student S1,
and logical student S2 — each as a separate instance.

Architecture (verified — see ADR-0001 corrected section):
    Stem: Conv(3->64, k=7, s=1, p=3) + ReLU + AvgPool(2)  -> 64 @ 128x128
    Block 1: 1x ResidualBlock(64, k=3) -> 1x1 conv(64->C)  -> X^1 @ C x 128x128
    Block 2: 2x ResidualBlock(64, k=3) -> 1x1 conv(64->C)  -> X^2 @ C x 128x128
    Block 3: 2x ResidualBlock(64, k=5) -> 1x1 conv(64->C)  -> X^3 @ C x 128x128

RF after each block: 16, 32, 64.
Parameters per network: 666,496 (2.67 MB) with default config.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from mudenet.models.common import ResidualBlock, Stem


class TeacherNetwork(nn.Module):
    """Multi-scale embedding network from paper Figure 2 (Sec. 3.1, Eq. 1).

    Produces L spatially-aligned embedding maps at increasing receptive fields.
    Used as both the teacher T and students S1 (Eq. 2), S2 (Eq. 6).

    The architecture is configurable: ``block_depths`` and ``kernel_sizes``
    control the number of residual blocks and kernel sizes per level,
    allowing receptive field sizes to be adapted (as noted in the paper).

    Args:
        internal_channels: Internal feature channel count. Default 64.
        output_channels: Output embedding channels C. Default 128.
        num_levels: Number of embedding levels L. Default 3.
        block_depths: Number of residual blocks per level. Default (1, 2, 2).
        kernel_sizes: Kernel size per level. Default (3, 3, 5).
    """

    def __init__(
        self,
        internal_channels: int = 64,
        output_channels: int = 128,
        num_levels: int = 3,
        block_depths: tuple[int, ...] = (1, 2, 2),
        kernel_sizes: tuple[int, ...] = (3, 3, 5),
    ) -> None:
        super().__init__()

        if len(block_depths) != num_levels:
            raise ValueError(
                f"block_depths length ({len(block_depths)}) must match "
                f"num_levels ({num_levels})"
            )
        if len(kernel_sizes) != num_levels:
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match "
                f"num_levels ({num_levels})"
            )

        self.num_levels = num_levels
        self.stem = Stem(out_channels=internal_channels)

        # Build residual blocks and 1x1 projection layers for each level.
        # Blocks are cumulative: block l operates on the output of block l-1.
        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()

        for level in range(num_levels):
            depth = block_depths[level]
            kernel = kernel_sizes[level]

            # Stack of residual blocks for this level
            block = nn.Sequential(
                *[ResidualBlock(internal_channels, kernel) for _ in range(depth)]
            )
            self.blocks.append(block)

            # 1x1 projection: internal_channels -> output_channels
            # Bare conv without BN/ReLU (see A-004 in assumptions-register.md)
            proj = nn.Conv2d(internal_channels, output_channels, kernel_size=1)
            self.projections.append(proj)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass producing multi-scale embedding maps (Eq. 1).

        Args:
            x: Input image tensor (B, 3, 256, 256).

        Returns:
            List of L embedding maps, each (B, C, 128, 128).
        """
        features = self.stem(x)  # (B, internal_channels, 128, 128)

        embedding_maps: list[Tensor] = []
        for level in range(self.num_levels):
            features = self.blocks[level](features)  # (B, internal_channels, 128, 128)
            projected = self.projections[level](features)  # (B, output_channels, 128, 128)
            embedding_maps.append(projected)

        return embedding_maps
