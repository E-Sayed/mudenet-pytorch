"""Tests for mudenet.visualization.overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mudenet.visualization.overlay import (
    create_visualization,
    denormalize_image,
    overlay_anomaly_map,
    save_visualization,
)

# Small image dimensions for fast tests
H, W = 32, 32


def _make_uint8_image(seed: int = 42) -> np.ndarray:
    """Create a random uint8 RGB image (H, W, 3)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(H, W, 3), dtype=np.uint8)


def _make_float_image(seed: int = 43) -> np.ndarray:
    """Create a random float RGB image (H, W, 3) in [0, 1]."""
    rng = np.random.RandomState(seed)
    return rng.rand(H, W, 3).astype(np.float32)


def _make_anomaly_map(seed: int = 44) -> np.ndarray:
    """Create a random anomaly score map (H, W)."""
    rng = np.random.RandomState(seed)
    return rng.rand(H, W).astype(np.float32)


# ---------------------------------------------------------------------------
# overlay_anomaly_map
# ---------------------------------------------------------------------------


class TestOverlayAnomalyMap:
    """Tests for overlay_anomaly_map."""

    def test_output_shape(self) -> None:
        """Output shape matches input image spatial dims."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map)
        assert result.shape == (H, W, 3)

    def test_output_dtype_uint8(self) -> None:
        """Output dtype is uint8."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map)
        assert result.dtype == np.uint8

    def test_output_range(self) -> None:
        """Output values are in [0, 255]."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_anomaly_map_zeros(self) -> None:
        """Uniform anomaly map (all zeros) produces valid output."""
        image = _make_uint8_image()
        anomaly_map = np.zeros((H, W), dtype=np.float32)
        result = overlay_anomaly_map(image, anomaly_map)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_uniform_anomaly_map_same_value(self) -> None:
        """Uniform anomaly map (all same non-zero value) produces valid output."""
        image = _make_uint8_image()
        anomaly_map = np.full((H, W), fill_value=5.0, dtype=np.float32)
        result = overlay_anomaly_map(image, anomaly_map)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_float_input_image(self) -> None:
        """Float [0, 1] input image produces valid output."""
        image = _make_float_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_uint8_input_image(self) -> None:
        """uint8 [0, 255] input image produces valid output."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_alpha_zero_returns_original(self) -> None:
        """alpha=0 returns the original image (no heatmap)."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map, alpha=0.0)
        np.testing.assert_array_equal(result, image)

    def test_alpha_one_returns_heatmap(self) -> None:
        """alpha=1 returns pure heatmap (no original image)."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map, alpha=1.0)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8
        # Result should actually differ from the original image
        assert not np.array_equal(result, image)

    def test_spatial_mismatch_raises(self) -> None:
        """Mismatched spatial dims raise ValueError."""
        image = _make_uint8_image()
        anomaly_map = np.zeros((H + 1, W + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Spatial dimensions mismatch"):
            overlay_anomaly_map(image, anomaly_map)

    def test_custom_colormap(self) -> None:
        """Custom colormap works without error."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = overlay_anomaly_map(image, anomaly_map, colormap="viridis")
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_2d_image_raises(self) -> None:
        """2D image (missing channel dim) raises ValueError."""
        image = np.zeros((H, W), dtype=np.uint8)
        anomaly_map = _make_anomaly_map()
        with pytest.raises(ValueError, match="Expected image shape"):
            overlay_anomaly_map(image, anomaly_map)

    def test_4_channel_image_raises(self) -> None:
        """4-channel image raises ValueError."""
        image = np.zeros((H, W, 4), dtype=np.uint8)
        anomaly_map = _make_anomaly_map()
        with pytest.raises(ValueError, match="Expected image shape"):
            overlay_anomaly_map(image, anomaly_map)

    def test_3d_anomaly_map_raises(self) -> None:
        """3D anomaly map raises ValueError."""
        image = _make_uint8_image()
        anomaly_map = np.zeros((1, H, W), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D anomaly map"):
            overlay_anomaly_map(image, anomaly_map)


# ---------------------------------------------------------------------------
# create_visualization
# ---------------------------------------------------------------------------


class TestCreateVisualization:
    """Tests for create_visualization."""

    def test_output_shape_without_gt(self) -> None:
        """Without ground truth, output width is 2*W."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = create_visualization(image, anomaly_map)
        assert result.shape == (H, W * 2, 3)

    def test_output_shape_with_gt(self) -> None:
        """With ground truth, output width is 3*W."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        rng = np.random.RandomState(50)
        gt_mask = (rng.rand(H, W) > 0.5).astype(np.uint8)
        result = create_visualization(image, anomaly_map, ground_truth_mask=gt_mask)
        assert result.shape == (H, W * 3, 3)

    def test_output_dtype(self) -> None:
        """Output dtype is uint8."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = create_visualization(image, anomaly_map)
        assert result.dtype == np.uint8

    def test_first_panel_is_original(self) -> None:
        """First panel in the visualization is the original image."""
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        result = create_visualization(image, anomaly_map)
        np.testing.assert_array_equal(result[:, :W, :], image)

    def test_gt_panel_has_red_tint(self) -> None:
        """Ground truth panel has red-tinted anomalous pixels."""
        image = np.full((H, W, 3), 128, dtype=np.uint8)
        anomaly_map = _make_anomaly_map()
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        gt_mask[0, 0] = 1  # One anomalous pixel

        result = create_visualization(image, anomaly_map, ground_truth_mask=gt_mask)
        # The GT panel is the second panel (index W:2*W)
        gt_panel = result[:, W : 2 * W, :]
        # The anomalous pixel should have more red than the original
        assert gt_panel[0, 0, 0] > gt_panel[0, 0, 1]  # R > G


# ---------------------------------------------------------------------------
# save_visualization
# ---------------------------------------------------------------------------


class TestSaveVisualization:
    """Tests for save_visualization."""

    def test_saves_valid_png(self, tmp_path: Path) -> None:
        """Writes a valid PNG file to disk."""
        path = tmp_path / "viz" / "test.png"
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()

        save_visualization(image, anomaly_map, output_path=path)

        assert path.exists()
        pil_img = Image.open(str(path))
        assert pil_img.mode == "RGB"
        assert pil_img.size == (W * 2, H)  # width, height
        pil_img.close()

    def test_saves_png_with_gt(self, tmp_path: Path) -> None:
        """Writes a valid PNG with ground truth panel."""
        path = tmp_path / "test_gt.png"
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()
        rng = np.random.RandomState(51)
        gt_mask = (rng.rand(H, W) > 0.5).astype(np.uint8)

        save_visualization(
            image, anomaly_map, output_path=path, ground_truth_mask=gt_mask
        )

        assert path.exists()
        pil_img = Image.open(str(path))
        assert pil_img.mode == "RGB"
        assert pil_img.size == (W * 3, H)
        pil_img.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Creates parent directories if they don't exist."""
        path = tmp_path / "deep" / "nested" / "dir" / "viz.png"
        image = _make_uint8_image()
        anomaly_map = _make_anomaly_map()

        save_visualization(image, anomaly_map, output_path=path)
        assert path.exists()


# ---------------------------------------------------------------------------
# denormalize_image
# ---------------------------------------------------------------------------


class TestDenormalizeImage:
    """Tests for denormalize_image."""

    def test_output_dtype_uint8(self) -> None:
        """Output dtype is uint8."""
        tensor = torch.randn(3, H, W)
        result = denormalize_image(tensor)
        assert result.dtype == np.uint8

    def test_output_range(self) -> None:
        """Output values are in [0, 255]."""
        tensor = torch.randn(3, H, W)
        result = denormalize_image(tensor)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_output_shape_chw(self) -> None:
        """(C, H, W) input produces (H, W, 3) output."""
        tensor = torch.randn(3, H, W)
        result = denormalize_image(tensor)
        assert result.shape == (H, W, 3)

    def test_output_shape_hwc(self) -> None:
        """(H, W, C) input produces (H, W, 3) output."""
        tensor = torch.randn(H, W, 3)
        result = denormalize_image(tensor)
        assert result.shape == (H, W, 3)

    def test_round_trip_approximate(self) -> None:
        """Normalize then denormalize approximately recovers the original.

        Due to uint8 quantization, we accept a tolerance of 2.
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # Create an original image in [0, 1]
        torch.manual_seed(42)
        original = torch.rand(3, H, W)

        # Normalize (simulating ImageNet normalization)
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        normalized = (original - mean_t) / std_t

        # Denormalize back
        recovered = denormalize_image(normalized, mean=mean, std=std)

        # Convert original to uint8 for comparison
        original_uint8 = (original.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Allow tolerance of 2 due to float → uint8 quantization
        np.testing.assert_allclose(
            recovered.astype(np.float32),
            original_uint8.astype(np.float32),
            atol=2.0,
        )

    def test_chw_and_hwc_same_result(self) -> None:
        """(C, H, W) and (H, W, C) layouts produce the same output."""
        torch.manual_seed(123)
        chw = torch.randn(3, H, W)
        hwc = chw.permute(1, 2, 0)

        result_chw = denormalize_image(chw)
        result_hwc = denormalize_image(hwc)

        np.testing.assert_array_equal(result_chw, result_hwc)

    def test_wrong_ndim_raises(self) -> None:
        """Non-3D tensor raises ValueError."""
        tensor = torch.randn(1, 3, H, W)
        with pytest.raises(ValueError, match="Expected 3-dim tensor"):
            denormalize_image(tensor)

    def test_wrong_channels_raises(self) -> None:
        """Tensor without 3-channel dim raises ValueError."""
        tensor = torch.randn(H, W, 4)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            denormalize_image(tensor)
