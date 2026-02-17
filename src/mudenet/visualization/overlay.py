"""Anomaly map visualization utilities.

Functions for overlaying anomaly heatmaps on original images, creating
side-by-side comparison panels, and saving visualizations to disk.

All visualization functions operate on numpy arrays. The one exception is
``denormalize_image``, which accepts a ``torch.Tensor`` and returns a
``np.ndarray`` — it serves as the tensor-to-numpy conversion boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert ImageNet-normalized tensor to uint8 numpy RGB image.

    Handles both channel-first ``(C, H, W)`` and channel-last ``(H, W, C)``
    layouts. The normalization is reversed with
    ``pixel = pixel * std + mean``, then values are clamped to [0, 1] and
    scaled to [0, 255].

    Args:
        image_tensor: Normalized image tensor, shape ``(3, H, W)`` or
            ``(H, W, 3)``.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        Denormalized image as uint8 array, shape ``(H, W, 3)``,
        range [0, 255].

    Raises:
        ValueError: If the tensor does not have exactly 3 dims or its
            channel dimension is not 3.
    """
    if image_tensor.ndim != 3:
        msg = (
            f"Expected 3-dim tensor (C, H, W) or (H, W, C), "
            f"got {image_tensor.ndim} dims."
        )
        raise ValueError(msg)

    arr = image_tensor.detach().cpu().float().numpy()

    # Detect layout: (C, H, W) vs (H, W, C)
    if arr.shape[0] == 3:
        # (C, H, W) or ambiguous (3, H, 3) — assume channel-first
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[2] != 3:
        msg = f"Expected 3 channels, got shape {arr.shape}."
        raise ValueError(msg)

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    # Reverse normalization: pixel = pixel * std + mean
    arr = arr * std_arr + mean_arr

    # Clamp to [0, 1] and scale to [0, 255]
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 [0, 255].

    Accepts both uint8 [0, 255] and float [0, 1] images.

    Args:
        image: RGB image, shape ``(H, W, 3)``.

    Returns:
        Image as uint8 array, shape ``(H, W, 3)``.

    Raises:
        ValueError: If the image does not have shape ``(H, W, 3)``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        msg = f"Expected image shape (H, W, 3), got {image.shape}."
        raise ValueError(msg)

    if image.dtype == np.uint8:
        return image

    # Float image — assume [0, 1]
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _normalize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    """Normalize anomaly map to [0, 1].

    Handles the edge case where all values are the same (uniform map)
    by returning a zero-intensity map.

    Args:
        anomaly_map: Anomaly score map, shape ``(H, W)``, any range.

    Returns:
        Normalized map in [0, 1] as float32.

    Raises:
        ValueError: If the anomaly map is not 2D.
    """
    if anomaly_map.ndim != 2:
        msg = f"Expected 2D anomaly map, got {anomaly_map.ndim} dims."
        raise ValueError(msg)

    map_float = anomaly_map.astype(np.float32)
    vmin = map_float.min()
    vmax = map_float.max()

    if vmax - vmin < 1e-8:
        # Uniform map — return zeros (no anomaly)
        return np.zeros_like(map_float)

    return (map_float - vmin) / (vmax - vmin)


def overlay_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay anomaly heatmap on original image.

    Args:
        image: Original RGB image, shape ``(H, W, 3)``, values in
            [0, 255] uint8 or [0, 1] float.
        anomaly_map: Anomaly score map, shape ``(H, W)``, any range
            (will be normalized to [0, 1]).
        colormap: Matplotlib colormap name. Default ``"jet"``.
        alpha: Blending factor for the heatmap overlay. Default 0.5.
            Blending formula:
            ``result = (1 - alpha) * image + alpha * heatmap``
            (convex combination, not additive).

    Returns:
        Blended image as uint8 array, shape ``(H, W, 3)``.

    Raises:
        ValueError: If image/anomaly_map shapes are invalid or spatial
            dimensions do not match.
    """
    img = _ensure_uint8_image(image)
    norm_map = _normalize_anomaly_map(anomaly_map)

    if img.shape[:2] != norm_map.shape:
        msg = (
            f"Spatial dimensions mismatch: image {img.shape[:2]} "
            f"vs anomaly_map {norm_map.shape}."
        )
        raise ValueError(msg)

    # Apply colormap: returns (H, W, 4) float RGBA in [0, 1]
    cmap = matplotlib.colormaps[colormap]
    heatmap_rgba = cmap(norm_map)  # (H, W, 4)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255.0).astype(np.uint8)  # (H, W, 3)

    # Convex combination: result = (1 - alpha) * image + alpha * heatmap
    blended = (
        (1.0 - alpha) * img.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    )
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _render_ground_truth_overlay(
    image: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> np.ndarray:
    """Render ground truth mask as red overlay on the original image.

    Anomalous pixels are tinted red; nominal pixels are unchanged.

    Args:
        image: Original RGB image, uint8 ``(H, W, 3)``.
        ground_truth_mask: Binary mask, shape ``(H, W)``. Values 0
            (nominal) or 1 (anomalous).

    Returns:
        Image with red overlay as uint8 ``(H, W, 3)``.
    """
    result = image.copy().astype(np.float32)
    mask_bool = ground_truth_mask.astype(bool)

    # Tint anomalous pixels red: blend with pure red
    red_tint = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    alpha_gt = 0.5
    result[mask_bool] = (1.0 - alpha_gt) * result[mask_bool] + alpha_gt * red_tint

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def create_visualization(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    ground_truth_mask: np.ndarray | None = None,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Create side-by-side visualization: input | [ground truth] | prediction.

    Produces a horizontal concatenation:

    - With ground truth: ``[original | GT mask | anomaly overlay]``
    - Without ground truth: ``[original | anomaly overlay]``

    Args:
        image: Original RGB image, shape ``(H, W, 3)``.
        anomaly_map: Anomaly score map, shape ``(H, W)``.
        ground_truth_mask: Optional binary ground-truth mask, shape
            ``(H, W)``. Values 0 (nominal) or 1 (anomalous).
        colormap: Matplotlib colormap name. Default ``"jet"``.
        alpha: Blending factor. Default 0.5.

    Returns:
        Side-by-side visualization as uint8 array, shape
        ``(H, W*N, 3)`` where N is 2 or 3 depending on whether
        ground truth is provided.

    Raises:
        ValueError: If shapes are invalid or incompatible.
    """
    img = _ensure_uint8_image(image)
    overlay = overlay_anomaly_map(img, anomaly_map, colormap=colormap, alpha=alpha)

    panels = [img]

    if ground_truth_mask is not None:
        gt_overlay = _render_ground_truth_overlay(img, ground_truth_mask)
        panels.append(gt_overlay)

    panels.append(overlay)

    return np.concatenate(panels, axis=1)


def save_visualization(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    output_path: str | Path,
    ground_truth_mask: np.ndarray | None = None,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> None:
    """Create and save visualization to disk as PNG.

    Creates parent directories automatically if they don't exist.

    Args:
        image: Original RGB image, shape ``(H, W, 3)``.
        anomaly_map: Anomaly score map, shape ``(H, W)``.
        output_path: Path to save the PNG file.
        ground_truth_mask: Optional binary ground-truth mask, shape
            ``(H, W)``.
        colormap: Matplotlib colormap name. Default ``"jet"``.
        alpha: Blending factor. Default 0.5.
    """
    vis = create_visualization(
        image=image,
        anomaly_map=anomaly_map,
        ground_truth_mask=ground_truth_mask,
        colormap=colormap,
        alpha=alpha,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pil_image = Image.fromarray(vis, mode="RGB")
    pil_image.save(str(out), format="PNG")
    logger.info("Saved visualization to %s", out)
