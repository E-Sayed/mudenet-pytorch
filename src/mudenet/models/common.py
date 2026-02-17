"""Shared building blocks for MuDeNet networks.

Contains the Stem and ResidualBlock modules used by the teacher
and student networks (Figure 2 in the paper).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """Two-convolution residual block from paper Figure 2.

    Architecture:
        Conv(k) -> BN -> ReLU -> Conv(k) -> BN -> (+skip) -> ReLU

    Uses same-padding: pad = (k - 1) // 2 to preserve spatial dimensions.
    The skip connection is an identity mapping (channels are preserved).

    See assumptions-register.md:
        A-001 (padding formula), A-003 (BN/ReLU placement).

    Args:
        channels: Number of input and output channels.
        kernel_size: Convolution kernel size (odd).
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for same-padding, got {kernel_size}"
            )
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual skip connection.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Output tensor (B, C, H, W) — same shape as input.
        """
        identity = x  # (B, C, H, W)

        out = self.conv1(x)   # (B, C, H, W)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)  # (B, C, H, W)
        out = self.bn2(out)

        out = out + identity   # Residual connection
        out = self.relu2(out)  # (B, C, H, W)
        return out


class Stem(nn.Module):
    """Teacher/student stem from paper Figure 2.

    Architecture:
        Conv2d(3, out_channels, kernel_size=7, stride=1, padding=3) -> ReLU -> AvgPool2d(2, 2)

    Reduces spatial dimensions from 256x256 to 128x128 while expanding
    to ``out_channels`` (default 64) feature channels.

    Receptive field after stem: 8, accumulated stride: 2.

    See assumptions-register.md:
        A-002 (pool type), A-005 (stem activation).

    Args:
        out_channels: Number of output channels. Default 64.
    """

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            3, out_channels, kernel_size=7, stride=1, padding=3, bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through stem.

        Args:
            x: Input image tensor (B, 3, 256, 256).

        Returns:
            Feature tensor (B, out_channels, 128, 128).
        """
        x = self.conv(x)   # (B, out_channels, 256, 256)
        x = self.relu(x)
        x = self.pool(x)   # (B, out_channels, 128, 128)
        return x
