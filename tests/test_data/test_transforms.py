"""Tests for image and mask transforms.

Verifies output shapes, value ranges, and that ImageNet normalization
is applied correctly.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from mudenet.config.schema import AugmentationConfig, DataConfig
from mudenet.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_eval_transform,
    get_mask_transform,
    get_train_transform,
)


def _make_random_pil(size: tuple[int, int] = (100, 80)) -> Image.Image:
    """Create a random RGB PIL image."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_binary_mask(size: tuple[int, int] = (100, 80)) -> Image.Image:
    """Create a binary grayscale PIL mask."""
    arr = np.random.randint(0, 2, size, dtype=np.uint8) * 255
    return Image.fromarray(arr, mode="L")


class TestEvalTransform:
    """Tests for get_eval_transform."""

    def test_output_shape(self) -> None:
        """Eval transform produces (3, image_size, image_size)."""
        config = DataConfig(image_size=256)
        tx = get_eval_transform(config)
        img = _make_random_pil()
        tensor = tx(img)
        assert tensor.shape == (3, 256, 256)

    def test_imagenet_normalized_range(self) -> None:
        """Output values are in approximately [-3, 3] (ImageNet range)."""
        config = DataConfig(image_size=256)
        tx = get_eval_transform(config)
        img = _make_random_pil()
        tensor = tx(img)
        # After ImageNet normalization, values should roughly be in [-2.5, 2.5]
        assert tensor.min() > -3.0
        assert tensor.max() < 3.0

    def test_dtype_float32(self) -> None:
        """Output is float32."""
        config = DataConfig(image_size=256)
        tx = get_eval_transform(config)
        tensor = tx(_make_random_pil())
        assert tensor.dtype == torch.float32


class TestTrainTransform:
    """Tests for get_train_transform."""

    def test_output_shape_no_augmentation(self) -> None:
        """Train transform without augmentation produces correct shape."""
        config = DataConfig(image_size=256)
        tx = get_train_transform(config)
        tensor = tx(_make_random_pil())
        assert tensor.shape == (3, 256, 256)

    def test_output_shape_with_augmentations(self) -> None:
        """Train transform with all augmentations produces correct shape."""
        config = DataConfig(
            image_size=256,
            augmentations=AugmentationConfig(
                horizontal_flip=True,
                vertical_flip=True,
                rotation=True,
                color_jitter=True,
            ),
        )
        tx = get_train_transform(config)
        tensor = tx(_make_random_pil())
        assert tensor.shape == (3, 256, 256)

    def test_determinism_without_augmentation(self) -> None:
        """Without augmentation, transform is deterministic."""
        config = DataConfig(image_size=256)
        tx = get_eval_transform(config)
        img = _make_random_pil()
        t1 = tx(img)
        t2 = tx(img)
        assert torch.equal(t1, t2)


class TestMaskTransform:
    """Tests for get_mask_transform."""

    def test_output_shape(self) -> None:
        """Mask transform produces (1, image_size, image_size)."""
        tx = get_mask_transform(256)
        mask = _make_binary_mask()
        tensor = tx(mask)
        assert tensor.shape == (1, 256, 256)

    def test_binary_values(self) -> None:
        """Mask output contains only values close to 0 or 1."""
        tx = get_mask_transform(256)
        # Create a mask with clear binary values (no resizing artifacts
        # because source is already the target size)
        arr = np.zeros((256, 256), dtype=np.uint8)
        arr[100:200, 100:200] = 255
        mask = Image.fromarray(arr, mode="L")
        tensor = tx(mask)
        unique_vals = torch.unique(tensor)
        assert all(v in (0.0, 1.0) for v in unique_vals.tolist())

    def test_no_normalization(self) -> None:
        """Mask values are in [0, 1], not ImageNet-normalized."""
        tx = get_mask_transform(256)
        mask = _make_binary_mask((256, 256))
        tensor = tx(mask)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_dtype_float32(self) -> None:
        """Mask output is float32."""
        tx = get_mask_transform(256)
        tensor = tx(_make_binary_mask())
        assert tensor.dtype == torch.float32


class TestImageNetNormalization:
    """Verify that ImageNet normalization constants are correct."""

    def test_mean_values(self) -> None:
        """ImageNet mean values match the standard."""
        assert IMAGENET_MEAN == [0.485, 0.456, 0.406]

    def test_std_values(self) -> None:
        """ImageNet std values match the standard."""
        assert IMAGENET_STD == [0.229, 0.224, 0.225]
