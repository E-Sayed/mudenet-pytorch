"""Per-level anomaly score computation (Eqs. 9-10).

Scoring functions compute spatial anomaly maps for localization.
Unlike training losses (Eqs. 3, 5, 7) which average over B/H/W to produce
scalars, scoring preserves spatial dimensions for pixel-level detection.

Scoring functions:
    - structural_score: Teacher vs structural student S1 (Eq. 9)
    - logical_score: Autoencoder vs logical student S2 (Eq. 10)
"""

from __future__ import annotations

from torch import Tensor


def _pairwise_score(
    maps_a: list[Tensor],
    maps_b: list[Tensor],
) -> list[Tensor]:
    """Squared difference summed over channels, per level.

    For each level l, computes:
        score^l = sum_c (A^l - B^l)^2  →  (B, H, W)

    Unlike the training loss helper (_pairwise_loss in losses.py), this
    function preserves spatial dimensions for anomaly localization instead
    of averaging over B/H/W.

    Args:
        maps_a: First set of embedding maps [L x (B, C, H, W)].
        maps_b: Second set of embedding maps [L x (B, C, H, W)].

    Returns:
        Per-level score maps [L x (B, H, W)].

    Raises:
        ValueError: If the number of levels doesn't match or is zero.
    """
    if len(maps_a) != len(maps_b):
        raise ValueError(
            f"Number of levels must match: got {len(maps_a)} and {len(maps_b)}"
        )
    if len(maps_a) == 0:
        raise ValueError("At least one level is required")

    return [
        (a - b).pow(2).sum(dim=1)  # sum over C → (B, H, W)
        for a, b in zip(maps_a, maps_b, strict=True)
    ]


def structural_score(
    teacher_maps: list[Tensor],
    student_maps: list[Tensor],
) -> list[Tensor]:
    """Structural anomaly score per level (Eq. 9).

    For each level l:
        S^l_str = sum_c (X^l_T - X^l_S1)^2  →  (B, H, W)

    This is the squared difference summed over the channel dimension.
    Unlike training losses (which average over B/H/W), scoring preserves
    spatial dimensions for anomaly localization.

    Args:
        teacher_maps: Teacher embedding maps [L x (B, C, H, W)].
        student_maps: Student S1 embedding maps [L x (B, C, H, W)].

    Returns:
        Per-level structural score maps [L x (B, H, W)].
    """
    return _pairwise_score(teacher_maps, student_maps)


def logical_score(
    autoencoder_maps: list[Tensor],
    student_maps: list[Tensor],
) -> list[Tensor]:
    """Logical anomaly score per level (Eq. 10).

    For each level l:
        S^l_log = sum_c (X^l_A - X^l_S2)^2  →  (B, H, W)

    Args:
        autoencoder_maps: Autoencoder reconstructed maps [L x (B, C, H, W)].
        student_maps: Student S2 embedding maps [L x (B, C, H, W)].

    Returns:
        Per-level logical score maps [L x (B, H, W)].
    """
    return _pairwise_score(autoencoder_maps, student_maps)
