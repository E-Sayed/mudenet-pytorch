"""WideResNet50 feature extraction for pre-knowledge distillation.

Extracts multi-scale features from a pretrained WideResNet50-2, fuses them
via upsampling and concatenation, then samples C channels to produce the
distillation target E (Eq. 13-15, Sec. 3.3).

Used only during Stage 1 (distillation) to train the teacher network.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models

logger = logging.getLogger(__name__)

# Epsilon for numerical stability in z-score normalization
_STD_EPS = 1e-8


class FeatureExtractor(nn.Module):
    """WideResNet50-2 feature extractor for distillation targets (Sec. 3.3).

    Extracts features from layers 1-3 of a pretrained WideResNet50-2,
    upsamples to a common spatial resolution, concatenates, and randomly
    samples C channels to produce the distillation target E.

    The extractor is always frozen (eval mode, no gradients).

    Args:
        output_channels: Number of channels C to sample. Default 128.
        backbone: Torchvision model name. Default "wide_resnet50_2".
        seed: Random seed for reproducible channel sampling. Default 42.

    Attributes:
        channel_indices: Tensor of sampled channel indices, shape (C,).
            Must be saved alongside the distilled teacher checkpoint
            for reproducibility.
    """

    def __init__(
        self,
        output_channels: int = 128,
        backbone: str = "wide_resnet50_2",
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.output_channels = output_channels

        # Load pretrained backbone
        if backbone != "wide_resnet50_2":
            raise ValueError(
                f"Only 'wide_resnet50_2' backbone is supported, got '{backbone}'"
            )

        wrn = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        # Extract sub-modules for feature extraction
        self.stem = nn.Sequential(wrn.conv1, wrn.bn1, wrn.relu, wrn.maxpool)
        self.layer1 = wrn.layer1  # -> (B, 256, 64, 64) for 256x256 input
        self.layer2 = wrn.layer2  # -> (B, 512, 32, 32)
        self.layer3 = wrn.layer3  # -> (B, 1024, 16, 16)

        # Total concatenated channels: 256 + 512 + 1024 = 1792
        total_channels = 256 + 512 + 1024

        # Randomly sample C channels from the 1792 concatenated channels (Eq. 15).
        # The indices are fixed per instance and must be saved with the checkpoint.
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.register_buffer(
            "channel_indices",
            torch.randperm(total_channels, generator=rng)[:output_channels],
        )

        # Freeze all parameters — this model is never trained.
        # Both requires_grad_(False) AND eval() are needed:
        # - requires_grad_(False) prevents gradient computation
        # - eval() switches BatchNorm to use pretrained running stats
        #   instead of batch stats (which would overwrite running_mean/var)
        self.requires_grad_(False)
        self.eval()
        logger.info(
            "FeatureExtractor initialized: %s -> %d channels (from %d)",
            backbone, output_channels, total_channels,
        )

    def train(self, mode: bool = True) -> FeatureExtractor:
        """Override to prevent accidental train mode activation.

        The feature extractor must always remain in eval mode so that
        BatchNorm layers use pretrained ImageNet running statistics.

        Args:
            mode: If True, raises RuntimeError. If False, no-op.

        Returns:
            Self.

        Raises:
            RuntimeError: If mode is True.
        """
        if mode:
            raise RuntimeError(
                "FeatureExtractor is always frozen — cannot set train mode"
            )
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        """Extract, fuse, and sample features from the input image (Eq. 13-15).

        Args:
            x: Input image tensor (B, 3, 256, 256), ImageNet-normalized.

        Returns:
            Distillation target E of shape (B, C, H1, W1) where H1, W1
            is the spatial size of layer1 output (64x64 for 256x256 input).
        """
        # Extract features from three stages
        stem_out = self.stem(x)
        f1 = self.layer1(stem_out)  # (B, 256, 64, 64)
        f2 = self.layer2(f1)        # (B, 512, 32, 32)
        f3 = self.layer3(f2)        # (B, 1024, 16, 16)

        # Upsample f2 and f3 to match f1's spatial resolution (Eq. 14).
        # Nearest-neighbor upsampling (Kronecker product with ones matrix).
        target_size = f1.shape[2:]  # (H1, W1)
        f2_up = F.interpolate(f2, size=target_size, mode="nearest")  # (B, 512, 64, 64)
        f3_up = F.interpolate(f3, size=target_size, mode="nearest")  # (B, 1024, 64, 64)

        # Concatenate along channel dimension (Eq. 14)
        e_hat = torch.cat([f1, f2_up, f3_up], dim=1)  # (B, 1792, 64, 64)

        # Sample C channels (Eq. 15)
        e = e_hat[:, self.channel_indices, :, :]  # (B, C, 64, 64)

        # Z-score normalize per sample, per channel (Eq. 15 follow-up).
        # NOTE (A-011): The paper says "z-score based on ImageNet statistics"
        # which could mean either (a) per-sample per-channel statistics computed
        # from the extracted features (our implementation), or (b) pre-computed
        # global mean/std from ImageNet. We use per-sample statistics because
        # the sampled channels vary per seed, making global stats impractical.
        # Revisit if distillation quality is poor. See assumptions-register.md.
        mean = e.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        std = e.std(dim=[2, 3], keepdim=True) + _STD_EPS  # (B, C, 1, 1)
        e = (e - mean) / std  # (B, C, 64, 64)

        return e

    @torch.inference_mode()
    def extract(self, x: Tensor) -> Tensor:
        """Convenience wrapper that ensures no-grad inference.

        Args:
            x: Input image tensor (B, 3, 256, 256), ImageNet-normalized.

        Returns:
            Distillation target E of shape (B, C, H1, W1).
        """
        return self.forward(x)
