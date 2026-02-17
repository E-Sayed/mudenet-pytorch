"""Loss functions for MuDeNet training.

All losses are pure computation — no model references, no state.
They take tensors and return scalar losses.

Loss functions:
    - distillation_loss: Teacher distillation from WideResNet50 (Eq. 16)
    - structural_loss: Teacher vs structural student S1 (Eq. 3)
    - autoencoder_loss: Teacher vs autoencoder A (Eq. 5)
    - logical_loss: Autoencoder vs logical student S2 (Eq. 7)
    - composite_loss: Combined end-to-end loss (Eq. 8)
"""

from __future__ import annotations

import torch
from torch import Tensor


def _pairwise_loss(
    maps_a: list[Tensor],
    maps_b: list[Tensor],
) -> Tensor:
    """Normalized squared Frobenius norm between paired embedding maps.

    For each level l, computes a normalized ||A^l - B^l||^2_F:
    the squared difference is summed over channels C and averaged
    (mean) over batch B, height H, and width W. The per-level
    losses are then averaged over L levels.

    Note: The paper's Frobenius norm (literal sum) differs only in
    absolute scale; the mean reduction makes the loss magnitude
    independent of batch size and spatial resolution. The learning
    rate compensates for the constant factor.

    Args:
        maps_a: First set of embedding maps [L x (B, C, H, W)].
        maps_b: Second set of embedding maps [L x (B, C, H, W)].

    Returns:
        Scalar loss averaged over L levels.

    Raises:
        ValueError: If the number of levels doesn't match.
    """
    if len(maps_a) != len(maps_b):
        raise ValueError(
            f"Number of levels must match: got {len(maps_a)} and {len(maps_b)}"
        )
    if len(maps_a) == 0:
        raise ValueError("At least one level is required")

    per_level = torch.stack([
        (a - b).pow(2).sum(dim=1).mean()  # sum over C, mean over B, H, W
        for a, b in zip(maps_a, maps_b, strict=True)
    ])
    return per_level.mean()


def distillation_loss(
    target: Tensor,
    teacher_maps: list[Tensor],
) -> Tensor:
    """Distillation loss: MSE between WideResNet50 target and teacher maps (Eq. 16).

    Each level's teacher output is compared against the same target E
    (the distillation target from the pretrained feature extractor).
    The loss is the squared Frobenius norm summed over channels and
    averaged over batch/spatial dimensions, then averaged over L levels.

    Args:
        target: Distillation target (B, C, H, W) — already upsampled to
            match teacher spatial resolution.
        teacher_maps: Teacher embedding maps [L x (B, C, H, W)].

    Returns:
        Scalar loss averaged over L levels.

    Raises:
        ValueError: If teacher_maps is empty.
    """
    if len(teacher_maps) == 0:
        raise ValueError("At least one teacher map is required")

    # All L levels are compared against the same target E
    per_level = torch.stack([
        (target - t).pow(2).sum(dim=1).mean()  # sum over C, mean over B, H, W
        for t in teacher_maps
    ])
    return per_level.mean()


def structural_loss(
    teacher_maps: list[Tensor],
    student_maps: list[Tensor],
) -> Tensor:
    """Structural loss: squared Frobenius norm between T and S1 (Eq. 3).

    Computes ||X^l_T - X^l_S1||^2_F per level, with sum over channels
    and mean over batch/spatial dimensions. Averaged over L levels.

    Args:
        teacher_maps: Teacher embedding maps [L x (B, C, H, W)].
        student_maps: Structural student S1 embedding maps [L x (B, C, H, W)].

    Returns:
        Scalar loss averaged over L levels.
    """
    return _pairwise_loss(teacher_maps, student_maps)


def autoencoder_loss(
    teacher_maps: list[Tensor],
    autoencoder_maps: list[Tensor],
) -> Tensor:
    """Autoencoder reconstruction loss: squared Frobenius norm T vs A (Eq. 5).

    Computes ||X^l_T - X^l_A||^2_F per level, with sum over channels
    and mean over batch/spatial dimensions. Averaged over L levels.

    Args:
        teacher_maps: Teacher embedding maps [L x (B, C, H, W)].
        autoencoder_maps: Autoencoder reconstructed maps [L x (B, C, H, W)].

    Returns:
        Scalar loss averaged over L levels.
    """
    return _pairwise_loss(teacher_maps, autoencoder_maps)


def logical_loss(
    autoencoder_maps: list[Tensor],
    student_maps: list[Tensor],
) -> Tensor:
    """Logical student loss: squared Frobenius norm A vs S2 (Eq. 7).

    Computes ||X^l_A - X^l_S2||^2_F per level, with sum over channels
    and mean over batch/spatial dimensions. Averaged over L levels.

    Args:
        autoencoder_maps: Autoencoder reconstructed maps [L x (B, C, H, W)].
        student_maps: Logical student S2 embedding maps [L x (B, C, H, W)].

    Returns:
        Scalar loss averaged over L levels.
    """
    return _pairwise_loss(autoencoder_maps, student_maps)


def composite_loss(
    structural: Tensor,
    autoencoder: Tensor,
    logical: Tensor,
) -> Tensor:
    """Composite end-to-end loss: L1 + LA + L2 (Eq. 8).

    Each sub-loss is already averaged over L levels, so no extra
    division by L is needed.

    Args:
        structural: Structural loss L1 (scalar).
        autoencoder: Autoencoder loss LA (scalar).
        logical: Logical student loss L2 (scalar).

    Returns:
        Scalar total loss.
    """
    return structural + autoencoder + logical
