"""Tests for dataset loaders using synthetic directory structures.

Creates temporary directories with small dummy PNG images to test
dataset discovery, sample counting, and return dict structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch import Tensor

from mudenet.config.schema import DataConfig
from mudenet.data.datasets import BaseAnomalyDataset, MVTecAD, MVTecLOCO, VisA
from mudenet.data.transforms import get_eval_transform, get_mask_transform


def _create_dummy_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a small random RGB PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _create_dummy_mask(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a small binary mask PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 2, size, dtype=np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


# ---------------------------------------------------------------------------
# MVTec AD fixtures and tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mvtec_ad_root(tmp_path: Path) -> Path:
    """Create a synthetic MVTec AD directory structure."""
    root = tmp_path / "mvtec_ad"
    cat = "bottle"

    # Train: 3 nominal images
    for i in range(3):
        _create_dummy_image(root / cat / "train" / "good" / f"img_{i:03d}.png")

    # Test: 2 nominal + 2 defective (broken_large)
    for i in range(2):
        _create_dummy_image(root / cat / "test" / "good" / f"img_{i:03d}.png")
    for i in range(2):
        _create_dummy_image(root / cat / "test" / "broken_large" / f"img_{i:03d}.png")
        _create_dummy_mask(root / cat / "ground_truth" / "broken_large" / f"img_{i:03d}_mask.png")

    return root


class TestMVTecAD:
    """Tests for MVTecAD dataset."""

    def test_train_sample_count(self, mvtec_ad_root: Path) -> None:
        """Train split discovers all nominal images."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="train")
        assert len(ds) == 3

    def test_test_sample_count(self, mvtec_ad_root: Path) -> None:
        """Test split discovers nominal + anomalous images."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="test")
        assert len(ds) == 4  # 2 good + 2 broken_large

    def test_return_dict_keys(self, mvtec_ad_root: Path) -> None:
        """Each sample returns a dict with the expected keys."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="train")
        sample = ds[0]
        assert set(sample.keys()) == {"image", "mask", "label", "path"}

    def test_train_labels_all_nominal(self, mvtec_ad_root: Path) -> None:
        """All train samples have label=0."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="train")
        for i in range(len(ds)):
            assert ds[i]["label"] == 0

    def test_train_masks_all_zero(self, mvtec_ad_root: Path) -> None:
        """Nominal samples have zero-valued mask tensors."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="train")
        for i in range(len(ds)):
            mask = ds[i]["mask"]
            assert isinstance(mask, Tensor)
            assert mask.sum().item() == 0.0

    def test_anomalous_label(self, mvtec_ad_root: Path) -> None:
        """Anomalous test samples have label=1."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="test")
        anomalous = [ds[i] for i in range(len(ds)) if ds[i]["label"] == 1]
        assert len(anomalous) == 2

    def test_with_transforms(self, mvtec_ad_root: Path) -> None:
        """Transforms produce correct tensor shapes."""
        config = DataConfig(image_size=256)
        tx = get_eval_transform(config)
        mask_tx = get_mask_transform(config.image_size)
        ds = MVTecAD(mvtec_ad_root, "bottle", split="test", transform=tx, target_transform=mask_tx)

        for i in range(len(ds)):
            sample = ds[i]
            assert sample["image"].shape == (3, 256, 256)
            if sample["label"] == 1:
                assert sample["mask"] is not None
                assert sample["mask"].shape == (1, 256, 256)

    def test_invalid_split_raises(self, mvtec_ad_root: Path) -> None:
        """Invalid split name raises ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            MVTecAD(mvtec_ad_root, "bottle", split="validation")

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        """Non-existent data root raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
            MVTecAD(tmp_path / "nonexistent", "bottle", split="train")

    def test_path_is_string(self, mvtec_ad_root: Path) -> None:
        """The 'path' field is a string."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="train")
        assert isinstance(ds[0]["path"], str)

    def test_mask_without_target_transform(self, mvtec_ad_root: Path) -> None:
        """Anomalous masks are returned as tensors even without target_transform."""
        ds = MVTecAD(mvtec_ad_root, "bottle", split="test")
        for i in range(len(ds)):
            sample = ds[i]
            if sample["label"] == 1:
                assert sample["mask"] is not None


# ---------------------------------------------------------------------------
# MVTec LOCO fixtures and tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mvtec_loco_root(tmp_path: Path) -> Path:
    """Create a synthetic MVTec LOCO directory structure."""
    root = tmp_path / "mvtec_loco"
    cat = "breakfast_box"

    # Train: 2 nominal
    for i in range(2):
        _create_dummy_image(root / cat / "train" / "good" / f"img_{i:03d}.png")

    # Validation: 1 nominal
    _create_dummy_image(root / cat / "validation" / "good" / "img_000.png")

    # Test: 1 good + 1 logical + 1 structural
    _create_dummy_image(root / cat / "test" / "good" / "img_000.png")
    _create_dummy_image(root / cat / "test" / "logical_anomalies" / "img_000.png")
    _create_dummy_mask(root / cat / "ground_truth" / "logical_anomalies" / "img_000.png")
    _create_dummy_image(root / cat / "test" / "structural_anomalies" / "img_000.png")
    _create_dummy_mask(root / cat / "ground_truth" / "structural_anomalies" / "img_000.png")

    return root


class TestMVTecLOCO:
    """Tests for MVTecLOCO dataset."""

    def test_train_sample_count(self, mvtec_loco_root: Path) -> None:
        """Train split discovers nominal images."""
        ds = MVTecLOCO(mvtec_loco_root, "breakfast_box", split="train")
        assert len(ds) == 2

    def test_validation_split(self, mvtec_loco_root: Path) -> None:
        """Validation split is supported (unlike MVTec AD)."""
        ds = MVTecLOCO(mvtec_loco_root, "breakfast_box", split="validation")
        assert len(ds) == 1

    def test_test_sample_count(self, mvtec_loco_root: Path) -> None:
        """Test split discovers all defect types."""
        ds = MVTecLOCO(mvtec_loco_root, "breakfast_box", split="test")
        assert len(ds) == 3  # 1 good + 1 logical + 1 structural

    def test_anomalous_labels(self, mvtec_loco_root: Path) -> None:
        """Anomalous test samples have label=1."""
        ds = MVTecLOCO(mvtec_loco_root, "breakfast_box", split="test")
        labels = [ds[i]["label"] for i in range(len(ds))]
        assert labels.count(0) == 1
        assert labels.count(1) == 2

    def test_with_transforms(self, mvtec_loco_root: Path) -> None:
        """Transforms produce correct tensor shapes."""
        config = DataConfig(image_size=256, dataset_type="mvtec_loco")
        tx = get_eval_transform(config)
        mask_tx = get_mask_transform(config.image_size)
        ds = MVTecLOCO(
            mvtec_loco_root, "breakfast_box", split="test",
            transform=tx, target_transform=mask_tx,
        )

        for i in range(len(ds)):
            sample = ds[i]
            assert sample["image"].shape == (3, 256, 256)

    def test_invalid_split_raises(self, mvtec_loco_root: Path) -> None:
        """Invalid split name raises ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            MVTecLOCO(mvtec_loco_root, "breakfast_box", split="val")


# ---------------------------------------------------------------------------
# VisA fixtures and tests
# ---------------------------------------------------------------------------

@pytest.fixture
def visa_root(tmp_path: Path) -> Path:
    """Create a synthetic VisA directory structure."""
    root = tmp_path / "visa"
    cat = "candle"

    # Train: 2 nominal
    for i in range(2):
        _create_dummy_image(root / cat / "train" / "good" / f"img_{i:03d}.png")

    # Test: 1 good + 2 bad
    _create_dummy_image(root / cat / "test" / "good" / "img_000.png")
    for i in range(2):
        _create_dummy_image(root / cat / "test" / "bad" / f"img_{i:03d}.png")
        _create_dummy_mask(root / cat / "ground_truth" / "bad" / f"img_{i:03d}.png")

    return root


class TestVisA:
    """Tests for VisA dataset."""

    def test_train_sample_count(self, visa_root: Path) -> None:
        """Train split discovers nominal images."""
        ds = VisA(visa_root, "candle", split="train")
        assert len(ds) == 2

    def test_test_sample_count(self, visa_root: Path) -> None:
        """Test split discovers good + bad images."""
        ds = VisA(visa_root, "candle", split="test")
        assert len(ds) == 3  # 1 good + 2 bad

    def test_anomalous_labels(self, visa_root: Path) -> None:
        """Bad images have label=1."""
        ds = VisA(visa_root, "candle", split="test")
        labels = [ds[i]["label"] for i in range(len(ds))]
        assert labels.count(0) == 1
        assert labels.count(1) == 2

    def test_masks_for_anomalous(self, visa_root: Path) -> None:
        """Anomalous samples have non-zero masks; nominal have zero masks."""
        config = DataConfig(image_size=256, dataset_type="visa")
        tx = get_eval_transform(config)
        mask_tx = get_mask_transform(config.image_size)
        ds = VisA(visa_root, "candle", split="test", transform=tx, target_transform=mask_tx)

        for i in range(len(ds)):
            sample = ds[i]
            assert isinstance(sample["mask"], Tensor)
            if sample["label"] == 1:
                assert sample["mask"].shape == (1, 256, 256)
            else:
                assert sample["mask"].sum().item() == 0.0

    def test_invalid_split_raises(self, visa_root: Path) -> None:
        """Invalid split name raises ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            VisA(visa_root, "candle", split="validation")


# ---------------------------------------------------------------------------
# Base class / shared behavior tests
# ---------------------------------------------------------------------------


class TestBaseAnomalyDataset:
    """Tests for shared BaseAnomalyDataset behavior."""

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        """A directory with no images raises ValueError."""
        # Create the directory structure but with no images
        (tmp_path / "mvtec_ad" / "bottle" / "train" / "good").mkdir(parents=True)
        with pytest.raises(ValueError, match="No images found"):
            MVTecAD(tmp_path / "mvtec_ad", "bottle", split="train")

    def test_missing_mask_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing mask for anomalous image logs a warning."""
        root = tmp_path / "visa"
        cat = "candle"

        # Train images (needed only because we test the test split)
        _create_dummy_image(root / cat / "train" / "good" / "img_000.png")

        # Test: anomalous image WITHOUT a corresponding mask
        _create_dummy_image(root / cat / "test" / "good" / "img_000.png")
        _create_dummy_image(root / cat / "test" / "bad" / "img_000.png")
        # No ground_truth directory created

        with caplog.at_level(logging.WARNING, logger="mudenet.data.datasets"):
            ds = VisA(root, cat, split="test")

        assert any("No mask found" in msg for msg in caplog.messages)
        # The sample should still exist with a zero-valued mask tensor
        anomalous = [ds[i] for i in range(len(ds)) if ds[i]["label"] == 1]
        assert len(anomalous) == 1
        assert isinstance(anomalous[0]["mask"], Tensor)
        assert anomalous[0]["mask"].sum().item() == 0.0

    def test_is_abstract(self) -> None:
        """BaseAnomalyDataset cannot be instantiated directly."""
        assert hasattr(BaseAnomalyDataset, "__abstractmethods__")

    def test_subclass_valid_splits(self) -> None:
        """Each subclass defines valid_splits."""
        assert MVTecAD.valid_splits == ("train", "test")
        assert MVTecLOCO.valid_splits == ("train", "validation", "test")
        assert VisA.valid_splits == ("train", "test")
