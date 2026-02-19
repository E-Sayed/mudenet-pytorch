"""Full inference pipeline: normalization, scoring, and fusion (Eqs. 9-12).

Orchestrates all 4 networks (T, S1, A, S2) to produce pixel-level and
image-level anomaly scores. Includes min-max normalization of per-branch
scores using statistics computed on a nominal validation set (Sec. 3.2).

Usage:
    norm_stats = compute_normalization_stats(teacher, s1, ae, s2, val_loader,
                                             device="cuda")
    anomaly_map = score_batch(teacher, s1, ae, s2, images, norm_stats)
    image_scores = compute_image_score(anomaly_map)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from mudenet.inference.scoring import logical_score, structural_score
from mudenet.models.autoencoder import Autoencoder
from mudenet.models.teacher import TeacherNetwork

logger = logging.getLogger(__name__)

#: Small epsilon to avoid division by zero in min-max normalization.
_NORM_EPS: float = 1e-8

#: Required keys in serialized NormalizationStats dicts.
_NORM_STATS_KEYS: frozenset[str] = frozenset(
    {"structural_min", "structural_max", "logical_min", "logical_max"}
)


@dataclass
class NormalizationStats:
    """Min-max normalization statistics for score fusion.

    Computed from a validation set of nominal images (Sec. 3.2).
    Stored per-level for both structural and logical branches.

    These statistics are serializable and should be saved alongside model
    checkpoints so inference can be done without recomputing them.

    Attributes:
        structural_min: Per-level min values [L floats].
        structural_max: Per-level max values [L floats].
        logical_min: Per-level min values [L floats].
        logical_max: Per-level max values [L floats].
    """

    structural_min: list[float]
    structural_max: list[float]
    logical_min: list[float]
    logical_max: list[float]

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize to a plain dict for checkpoint storage.

        Returns:
            Dictionary with all four lists keyed by attribute name.
        """
        return {
            "structural_min": self.structural_min,
            "structural_max": self.structural_max,
            "logical_min": self.logical_min,
            "logical_max": self.logical_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> NormalizationStats:
        """Deserialize from a plain dict (loaded from checkpoint).

        Validates that all required keys are present, list lengths are
        consistent, and all values are finite.

        Args:
            data: Dictionary with keys structural_min, structural_max,
                logical_min, logical_max.

        Returns:
            NormalizationStats instance.

        Raises:
            ValueError: If required keys are missing, list lengths are
                inconsistent, or values are non-finite.
        """
        missing = _NORM_STATS_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required keys in NormalizationStats dict: {missing}"
            )

        lengths = {k: len(data[k]) for k in _NORM_STATS_KEYS}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"Inconsistent list lengths in NormalizationStats dict: {lengths}"
            )

        for key in _NORM_STATS_KEYS:
            for i, val in enumerate(data[key]):
                if not math.isfinite(val):
                    raise ValueError(
                        f"Non-finite value in NormalizationStats.{key}[{i}]: {val}"
                    )

        return cls(
            structural_min=data["structural_min"],
            structural_max=data["structural_max"],
            logical_min=data["logical_min"],
            logical_max=data["logical_max"],
        )


def compute_normalization_stats(
    teacher: TeacherNetwork,
    student1: TeacherNetwork,
    autoencoder: Autoencoder,
    student2: TeacherNetwork,
    dataloader: DataLoader,
    device: str | torch.device,
) -> NormalizationStats:
    """Compute min/max normalization statistics on nominal validation data.

    Runs all 4 networks on each batch, computes structural and logical
    scores, and tracks per-level min/max across the entire validation set.

    All models should already be on the correct device. This function puts
    them in eval mode. All forward passes run under torch.inference_mode().

    Args:
        teacher: Frozen teacher network.
        student1: Trained structural student S1.
        autoencoder: Trained autoencoder A.
        student2: Trained logical student S2.
        dataloader: Validation data loader (nominal images only).
        device: Compute device for moving input images.

    Returns:
        NormalizationStats with per-level min/max for both branches.
    """
    teacher.eval()
    student1.eval()
    autoencoder.eval()
    student2.eval()

    # Running min/max trackers — initialized on first batch when L is known
    struct_min: list[float] | None = None
    struct_max: list[float] | None = None
    logic_min: list[float] | None = None
    logic_max: list[float] | None = None

    progress = tqdm(dataloader, desc="Computing normalization stats", leave=False)

    with torch.inference_mode():
        for batch in progress:
            images = batch["image"].to(device)  # (B, 3, 256, 256)

            teacher_maps = teacher(images)          # L x (B, C, H, W)
            student1_maps = student1(images)        # L x (B, C, H, W)
            autoencoder_maps = autoencoder(images)  # L x (B, C, H, W)
            student2_maps = student2(images)        # L x (B, C, H, W)

            struct_scores = structural_score(teacher_maps, student1_maps)
            logic_scores = logical_score(autoencoder_maps, student2_maps)

            num_levels = len(struct_scores)

            # Initialize trackers on first batch
            if struct_min is None:
                struct_min = [float("inf")] * num_levels
                struct_max = [float("-inf")] * num_levels
                logic_min = [float("inf")] * num_levels
                logic_max = [float("-inf")] * num_levels

            # Update running min/max per level (global scalar over entire batch)
            assert (
                struct_max is not None
                and logic_min is not None
                and logic_max is not None
            )
            for level in range(num_levels):
                struct_min[level] = min(
                    struct_min[level], struct_scores[level].min().item()
                )
                struct_max[level] = max(
                    struct_max[level], struct_scores[level].max().item()
                )
                logic_min[level] = min(
                    logic_min[level], logic_scores[level].min().item()
                )
                logic_max[level] = max(
                    logic_max[level], logic_scores[level].max().item()
                )

    if struct_min is None:
        raise ValueError(
            "Dataloader yielded zero batches — cannot compute normalization stats"
        )

    assert (
        struct_max is not None and logic_min is not None and logic_max is not None
    )
    logger.info(
        "Normalization stats computed over %d batches (%d levels)",
        len(dataloader),
        len(struct_min),
    )

    return NormalizationStats(
        structural_min=struct_min,
        structural_max=struct_max,
        logical_min=logic_min,
        logical_max=logic_max,
    )


def score_batch(
    teacher: TeacherNetwork,
    student1: TeacherNetwork,
    autoencoder: Autoencoder,
    student2: TeacherNetwork,
    images: Tensor,
    norm_stats: NormalizationStats,
    *,
    clamp_scores: bool = False,
) -> Tensor:
    """Compute normalized anomaly map for a batch of images (Eqs. 9-12).

    Pipeline:
      1. Forward pass through all 4 networks
      2. Compute structural_score (Eq. 9) and logical_score (Eq. 10)
      3. Min-max normalize per level using norm_stats
      4. Per-level fusion: S^l = S^l_str_norm + S^l_log_norm  (Eq. 11)
      5. Average across levels: S = (1/L) * sum_l S^l  (Eq. 12)

    All models should already be on the correct device.
    Images should already be on the correct device.

    Args:
        teacher: Frozen teacher network.
        student1: Trained structural student S1.
        autoencoder: Trained autoencoder A.
        student2: Trained logical student S2.
        images: Input images (B, 3, 256, 256) — already on correct device.
        norm_stats: Pre-computed normalization statistics.
        clamp_scores: If True, clamp normalized per-branch scores to [0, 1]
            before fusion. Prevents test-time outliers from one branch
            dominating the fused map.

    Returns:
        Anomaly score map (B, H, W) — higher = more anomalous.

    Raises:
        ValueError: If norm_stats levels don't match model output levels.
    """
    teacher.eval()
    student1.eval()
    autoencoder.eval()
    student2.eval()

    with torch.inference_mode():
        teacher_maps = teacher(images)          # L x (B, C, H, W)
        student1_maps = student1(images)        # L x (B, C, H, W)
        autoencoder_maps = autoencoder(images)  # L x (B, C, H, W)
        student2_maps = student2(images)        # L x (B, C, H, W)

        struct_scores = structural_score(teacher_maps, student1_maps)
        logic_scores = logical_score(autoencoder_maps, student2_maps)

        num_levels = len(struct_scores)

        # Validate norm_stats level count
        if len(norm_stats.structural_min) != num_levels:
            raise ValueError(
                f"norm_stats has {len(norm_stats.structural_min)} levels "
                f"but model produced {num_levels} levels"
            )

        # Min-max normalize per level, per branch (Sec. 3.2)
        # Then fuse per level: S^l = S^l_str_norm + S^l_log_norm (Eq. 11)
        fused_levels: list[Tensor] = []
        for level in range(num_levels):
            struct_norm = _min_max_normalize(
                struct_scores[level],
                norm_stats.structural_min[level],
                norm_stats.structural_max[level],
            )
            logic_norm = _min_max_normalize(
                logic_scores[level],
                norm_stats.logical_min[level],
                norm_stats.logical_max[level],
            )
            if clamp_scores:
                struct_norm = struct_norm.clamp(0.0, 1.0)
                logic_norm = logic_norm.clamp(0.0, 1.0)
            fused_levels.append(struct_norm + logic_norm)  # Eq. 11: (B, H, W)

        # Average across levels: S = (1/L) * sum_l S^l (Eq. 12)
        anomaly_map = torch.stack(fused_levels).mean(dim=0)  # (B, H, W)

    return anomaly_map


def compute_image_score(anomaly_map: Tensor) -> Tensor:
    """Image-level anomaly score: max of the anomaly map (Sec. 3.2).

    Takes the spatial maximum of each sample's anomaly map to produce
    a single scalar score per image.

    Args:
        anomaly_map: Pixel-level score map (B, H, W).

    Returns:
        Image-level scores (B,).
    """
    return anomaly_map.flatten(start_dim=1).max(dim=1).values


def gaussian_smooth(anomaly_map: Tensor, sigma: float) -> Tensor:
    """Apply Gaussian smoothing to anomaly maps.

    Reduces pixel-level noise in the anomaly map, which improves
    image-level scoring (max is less sensitive to single-pixel spikes)
    and region-level metrics (PRO connected-component analysis).

    The kernel size is chosen to cover 4 standard deviations on each
    side, ensuring negligible truncation error.

    Args:
        anomaly_map: Pixel-level score map (B, H, W).
        sigma: Gaussian standard deviation in pixels.  Must be > 0.
            Typical value: 4.0 for 256x256 images.

    Returns:
        Smoothed anomaly map (B, H, W), same shape as input.

    Raises:
        ValueError: If sigma is not positive.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
    padding = kernel_size // 2

    # Build 1-D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=anomaly_map.device)
    coords -= kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d /= gauss_1d.sum()

    # Outer product → 2-D kernel, shaped (1, 1, K, K) for depthwise conv
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    # Conv2d expects (B, C, H, W) — add/remove channel dim
    x = anomaly_map.unsqueeze(1)  # (B, 1, H, W)
    x = torch.nn.functional.pad(x, [padding] * 4, mode="reflect")
    x = torch.nn.functional.conv2d(x, kernel)
    return x.squeeze(1)  # (B, H, W)


def _min_max_normalize(
    score: Tensor,
    min_val: float,
    max_val: float,
) -> Tensor:
    """Min-max normalize a score map using pre-computed statistics.

    Applies: normalized = (score - min) / (max - min + eps)

    Args:
        score: Raw score map (B, H, W).
        min_val: Global minimum from validation set (scalar).
        max_val: Global maximum from validation set (scalar).

    Returns:
        Normalized score map (B, H, W).
    """
    return (score - min_val) / (max_val - min_val + _NORM_EPS)
