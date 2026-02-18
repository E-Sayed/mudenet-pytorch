"""Evaluation metrics for anomaly detection and segmentation.

All metric functions are pure computation — no file I/O, no model references,
no torch tensors. They take numpy arrays and return scalar scores.

Metrics:
    - image_auroc: Image-level AUROC (binary classification)
    - pixel_auroc: Pixel-level AUROC (segmentation)
    - pro_score: Per-Region Overlap (PRO) — primary segmentation metric
    - spro_score: Saturated PRO — for MVTec LOCO evaluation (Sec. 4)
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from sklearn.metrics import auc, roc_auc_score


def image_auroc(
    labels: NDArray[np.integer],
    scores: NDArray[np.floating],
) -> float:
    """Image-level AUROC using sklearn.

    Args:
        labels: Ground truth binary labels (N,). 0 = nominal, 1 = anomalous.
        scores: Predicted anomaly scores (N,). Higher = more anomalous.

    Returns:
        Area under the ROC curve [0, 1].
        Returns 0.0 if all labels are the same class (undefined AUROC).

    Raises:
        ValueError: If labels and scores have different lengths or are empty.
    """
    _validate_1d_arrays(labels, scores, "labels", "scores")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        warnings.warn(
            f"Only one class present in labels ({unique_labels}). "
            "AUROC is undefined; returning 0.0.",
            stacklevel=2,
        )
        return 0.0

    return float(roc_auc_score(labels, scores))


def pixel_auroc(
    masks: NDArray[np.integer],
    anomaly_maps: NDArray[np.floating],
) -> float:
    """Pixel-level AUROC.

    Flattens masks and anomaly maps, then computes ROC AUC.

    Args:
        masks: Ground truth binary masks (N, H, W). 1 = anomalous pixel.
        anomaly_maps: Predicted anomaly score maps (N, H, W).

    Returns:
        Area under the pixel-level ROC curve [0, 1].
        Returns 0.0 if all mask values are the same (undefined AUROC).

    Raises:
        ValueError: If masks and anomaly_maps have different shapes or are empty.
    """
    _validate_spatial_arrays(masks, anomaly_maps, "masks", "anomaly_maps")

    flat_masks = masks.ravel()
    flat_scores = anomaly_maps.ravel()

    unique_labels = np.unique(flat_masks)
    if len(unique_labels) < 2:
        warnings.warn(
            f"Only one class present in masks ({unique_labels}). "
            "Pixel AUROC is undefined; returning 0.0.",
            stacklevel=2,
        )
        return 0.0

    return float(roc_auc_score(flat_masks, flat_scores))


def pro_score(
    masks: NDArray[np.integer],
    anomaly_maps: NDArray[np.floating],
    max_fpr: float = 0.3,
    num_thresholds: int = 1000,
) -> float:
    """Per-Region Overlap (PRO) score.

    For each threshold:
      1. Binarize anomaly_maps at that threshold
      2. For each connected component in the ground truth mask,
         compute the overlap (intersection / component area)
      3. PRO at this threshold = mean overlap across all components

    Then compute the area under the PRO-vs-FPR curve up to max_fpr,
    normalized to [0, 1] by dividing by max_fpr.

    Uses scipy.ndimage.label for connected component extraction.

    Args:
        masks: Ground truth binary masks (N, H, W). 1 = anomalous.
        anomaly_maps: Predicted score maps (N, H, W).
        max_fpr: Maximum false positive rate for integration. Default 0.3.
        num_thresholds: Number of thresholds to evaluate. Default 1000.

    Returns:
        Normalized area under the PRO curve up to max_fpr.

    Raises:
        ValueError: If masks and anomaly_maps have different shapes or are empty.
        ValueError: If max_fpr is not in (0, 1] or num_thresholds < 2.
    """
    _validate_spatial_arrays(masks, anomaly_maps, "masks", "anomaly_maps")
    _validate_pro_params(max_fpr, num_thresholds)

    return _compute_pro(
        masks, anomaly_maps, max_fpr, num_thresholds, saturation_threshold=None
    )


def spro_score(
    masks: NDArray[np.integer],
    anomaly_maps: NDArray[np.floating],
    saturation_threshold: float = 1.0,
    max_fpr: float = 0.05,
    num_thresholds: int = 1000,
) -> float:
    """Saturated Per-Region Overlap (sPRO) score for MVTec LOCO (Sec. 4).

    Same as PRO but per-component overlap is clamped to
    min(overlap, saturation_threshold). This prevents large
    components from dominating the score.

    Area under the sPRO-vs-FPR curve up to max_fpr,
    normalized by dividing by (saturation_threshold * max_fpr)
    so the result is in [0, 1].

    Note:
        This is a simplified sPRO that uses a fixed global saturation
        threshold and connected-component analysis. The official MVTec
        LOCO evaluation uses per-defect-type saturation areas loaded
        from ``defects_config.json``. See assumption A-015 in the
        assumptions register for details.

    Args:
        masks: Ground truth binary masks (N, H, W).
        anomaly_maps: Predicted score maps (N, H, W).
        saturation_threshold: Maximum overlap value per component. Default 1.0.
        max_fpr: Maximum FPR for integration. Default 0.05.
        num_thresholds: Number of thresholds. Default 1000.

    Returns:
        Normalized area under the sPRO curve up to max_fpr, in [0, 1].

    Raises:
        ValueError: If masks and anomaly_maps have different shapes or are empty.
        ValueError: If saturation_threshold <= 0, max_fpr not in (0, 1],
            or num_thresholds < 2.
    """
    _validate_spatial_arrays(masks, anomaly_maps, "masks", "anomaly_maps")
    _validate_pro_params(max_fpr, num_thresholds)
    if saturation_threshold <= 0:
        raise ValueError(
            f"saturation_threshold must be > 0, got {saturation_threshold}"
        )

    return _compute_pro(
        masks, anomaly_maps, max_fpr, num_thresholds, saturation_threshold
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_pro(
    masks: NDArray[np.integer],
    anomaly_maps: NDArray[np.floating],
    max_fpr: float,
    num_thresholds: int,
    saturation_threshold: float | None,
) -> float:
    """Shared PRO / sPRO computation.

    Args:
        masks: Ground truth binary masks (N, H, W).
        anomaly_maps: Predicted score maps (N, H, W).
        max_fpr: Maximum FPR for integration.
        num_thresholds: Number of thresholds to evaluate.
        saturation_threshold: If not None, clamp per-component overlap
            (sPRO mode). If None, standard PRO.

    Returns:
        Normalized area under the (s)PRO-vs-FPR curve.
    """
    # Pre-compute connected components for each image with anomalous GT
    components_per_image = _extract_connected_components(masks)

    if not components_per_image:
        # No anomalous ground truth — PRO is undefined
        warnings.warn(
            "No anomalous ground truth regions found. "
            "PRO score is undefined; returning 0.0.",
            stacklevel=3,
        )
        return 0.0

    # Total number of negative (non-anomalous) pixels for FPR computation
    total_negative = int(np.sum(masks == 0))
    if total_negative == 0:
        warnings.warn(
            "All pixels are anomalous in ground truth. "
            "FPR is undefined; returning 0.0.",
            stacklevel=3,
        )
        return 0.0

    # Linearly spaced thresholds from max to min score
    score_min = float(anomaly_maps.min())
    score_max = float(anomaly_maps.max())
    thresholds = np.linspace(score_max, score_min, num_thresholds)

    fpr_list: list[float] = []
    pro_list: list[float] = []

    for threshold in thresholds:
        pred = anomaly_maps >= threshold  # (N, H, W) binary

        # FPR = FP / (FP + TN) = FP / total_negative
        false_positives = int(np.sum(pred & (masks == 0)))
        fpr = false_positives / total_negative

        # Per-component overlap using labeled arrays
        overlaps: list[float] = []
        for img_idx, components in components_per_image:
            pred_img = pred[img_idx]
            for label_id, comp_slice, labeled_arr in components:
                # Use the labeled array to isolate this specific component
                component_mask = labeled_arr[comp_slice] == label_id
                component_area = int(np.sum(component_mask))
                if component_area == 0:
                    continue
                overlap = float(
                    np.sum(pred_img[comp_slice] & component_mask) / component_area
                )
                if saturation_threshold is not None:
                    overlap = min(overlap, saturation_threshold)
                overlaps.append(overlap)

        pro_at_threshold = float(np.mean(overlaps)) if overlaps else 0.0

        fpr_list.append(fpr)
        pro_list.append(pro_at_threshold)

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)

    # Normalize: divide by max_fpr for PRO, by sat * max_fpr for sPRO
    normalizer = max_fpr
    if saturation_threshold is not None:
        normalizer = saturation_threshold * max_fpr

    # Filter to FPR <= max_fpr and interpolate boundary
    return _integrate_pro_curve(fpr_arr, pro_arr, max_fpr, normalizer)


def _extract_connected_components(
    masks: NDArray[np.integer],
) -> list[tuple[int, list[tuple[int, tuple[slice, ...], NDArray[np.integer]]]]]:
    """Extract connected components from ground truth masks.

    Returns the labeled array alongside each component's bounding-box slice
    so that the caller can reconstruct the exact per-label mask (not just
    the bounding box).

    Args:
        masks: Ground truth binary masks (N, H, W).

    Returns:
        List of (image_index, components) for images with anomalies.
        Each components entry is a list of (label_id, slice_tuple, labeled_array).
    """
    result: list[
        tuple[int, list[tuple[int, tuple[slice, ...], NDArray[np.integer]]]]
    ] = []
    for i in range(masks.shape[0]):
        if np.sum(masks[i]) == 0:
            continue
        labeled, num_features = ndimage.label(masks[i])
        if num_features == 0:
            continue
        objects = ndimage.find_objects(labeled)
        components: list[tuple[int, tuple[slice, ...], NDArray[np.integer]]] = []
        for label_id, obj in enumerate(objects, start=1):
            if obj is not None:
                components.append((label_id, obj, labeled))
        if components:
            result.append((i, components))
    return result


def _integrate_pro_curve(
    fpr: NDArray[np.floating],
    pro: NDArray[np.floating],
    max_fpr: float,
    normalizer: float,
) -> float:
    """Integrate PRO curve up to max_fpr, normalized by the given divisor.

    Sorts by FPR, clips to [0, max_fpr], interpolates the boundary
    point, and integrates with the trapezoidal rule.

    Args:
        fpr: FPR values (K,).
        pro: PRO values (K,).
        max_fpr: Maximum FPR for integration.
        normalizer: Divisor for the raw area. For standard PRO this
            equals max_fpr; for sPRO it equals
            saturation_threshold * max_fpr.

    Returns:
        Normalized area under the curve in [0, 1].
    """
    # Sort by FPR (ascending)
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    pro_sorted = pro[sort_idx]

    # Filter to FPR <= max_fpr
    valid = fpr_sorted <= max_fpr
    fpr_clipped = fpr_sorted[valid]
    pro_clipped = pro_sorted[valid]

    # Interpolate boundary point at max_fpr if needed
    if len(fpr_clipped) == 0:
        return 0.0

    if fpr_clipped[-1] < max_fpr:
        # Interpolate PRO value at max_fpr from the closest points
        idx_above = np.searchsorted(fpr_sorted, max_fpr, side="left")
        if idx_above < len(fpr_sorted):
            # Linear interpolation between last valid and first above
            fpr_lo = fpr_clipped[-1]
            fpr_hi = fpr_sorted[idx_above]
            pro_lo = pro_clipped[-1]
            pro_hi = pro_sorted[idx_above]
            if fpr_hi > fpr_lo:
                alpha = (max_fpr - fpr_lo) / (fpr_hi - fpr_lo)
                pro_boundary = pro_lo + alpha * (pro_hi - pro_lo)
            else:
                pro_boundary = pro_lo
            fpr_clipped = np.append(fpr_clipped, max_fpr)
            pro_clipped = np.append(pro_clipped, pro_boundary)
        else:
            # All FPR values below max_fpr — extend with last PRO value
            fpr_clipped = np.append(fpr_clipped, max_fpr)
            pro_clipped = np.append(pro_clipped, pro_clipped[-1])

    if len(fpr_clipped) < 2:
        return 0.0

    area = float(auc(fpr_clipped, pro_clipped))
    return area / normalizer


def _validate_1d_arrays(
    a: NDArray, b: NDArray, name_a: str, name_b: str
) -> None:
    """Validate that two 1D arrays have matching non-zero length."""
    if a.ndim != 1:
        raise ValueError(f"{name_a} must be 1D, got shape {a.shape}")
    if b.ndim != 1:
        raise ValueError(f"{name_b} must be 1D, got shape {b.shape}")
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} must have the same length: "
            f"got {len(a)} and {len(b)}"
        )
    if len(a) == 0:
        raise ValueError(f"{name_a} and {name_b} must not be empty")


def _validate_spatial_arrays(
    a: NDArray, b: NDArray, name_a: str, name_b: str
) -> None:
    """Validate that two 3D (N, H, W) arrays have matching shapes."""
    if a.ndim != 3:
        raise ValueError(f"{name_a} must be 3D (N, H, W), got shape {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"{name_b} must be 3D (N, H, W), got shape {b.shape}")
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape: "
            f"got {a.shape} and {b.shape}"
        )
    if a.shape[0] == 0:
        raise ValueError(f"{name_a} and {name_b} must have at least one sample")


def _validate_pro_params(max_fpr: float, num_thresholds: int) -> None:
    """Validate PRO/sPRO parameters."""
    if not (0 < max_fpr <= 1):
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")
    if num_thresholds < 2:
        raise ValueError(f"num_thresholds must be >= 2, got {num_thresholds}")
