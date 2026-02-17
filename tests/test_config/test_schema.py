"""Tests for configuration dataclass validation."""

from __future__ import annotations

import pytest

from mudenet.config.schema import (
    AugmentationConfig,
    Config,
    DataConfig,
    DistillationConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_default_values(self) -> None:
        """Default ModelConfig creates successfully."""
        cfg = ModelConfig()
        assert cfg.num_channels == 128
        assert cfg.latent_dim == 32
        assert cfg.num_levels == 3
        assert cfg.internal_channels == 64
        assert cfg.block_depths == (1, 2, 2)
        assert cfg.kernel_sizes == (3, 3, 5)
        assert cfg.image_size == 256

    def test_block_depths_length_mismatch(self) -> None:
        """Mismatched block_depths length raises ValueError."""
        with pytest.raises(ValueError, match="block_depths length"):
            ModelConfig(num_levels=3, block_depths=(1, 2))

    def test_kernel_sizes_length_mismatch(self) -> None:
        """Mismatched kernel_sizes length raises ValueError."""
        with pytest.raises(ValueError, match="kernel_sizes length"):
            ModelConfig(num_levels=3, kernel_sizes=(3, 3))

    def test_num_channels_nonpositive(self) -> None:
        """Non-positive num_channels raises ValueError."""
        with pytest.raises(ValueError, match="num_channels must be positive"):
            ModelConfig(num_channels=0)

    def test_latent_dim_nonpositive(self) -> None:
        """Non-positive latent_dim raises ValueError."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            ModelConfig(latent_dim=-1)

    def test_image_size_not_256(self) -> None:
        """Non-256 image_size raises ValueError (encoder/decoder constraint)."""
        with pytest.raises(ValueError, match="image_size must be 256"):
            ModelConfig(image_size=512)

    def test_image_size_not_256_smaller(self) -> None:
        """Smaller image_size also raises ValueError."""
        with pytest.raises(ValueError, match="image_size must be 256"):
            ModelConfig(image_size=128)


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_default_values(self) -> None:
        """Default TrainingConfig creates successfully."""
        cfg = TrainingConfig()
        assert cfg.num_epochs == 500
        assert cfg.optimizer == "adam"

    def test_num_epochs_nonpositive(self) -> None:
        """Non-positive num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TrainingConfig(num_epochs=0)

    def test_batch_size_nonpositive(self) -> None:
        """Non-positive batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=-1)

    def test_learning_rate_nonpositive(self) -> None:
        """Non-positive learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0.0)

    def test_invalid_optimizer(self) -> None:
        """Invalid optimizer name raises ValueError."""
        with pytest.raises(ValueError, match="optimizer must be one of"):
            TrainingConfig(optimizer="sgd")

    def test_valid_optimizer_adam(self) -> None:
        """Adam optimizer is accepted."""
        cfg = TrainingConfig(optimizer="adam")
        assert cfg.optimizer == "adam"


class TestDistillationConfig:
    """Tests for DistillationConfig validation."""

    def test_default_values(self) -> None:
        """Default DistillationConfig creates successfully."""
        cfg = DistillationConfig()
        assert cfg.backbone == "wide_resnet50_2"
        assert cfg.learning_rate == 1e-3

    def test_num_epochs_nonpositive(self) -> None:
        """Non-positive num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            DistillationConfig(num_epochs=0)

    def test_batch_size_nonpositive(self) -> None:
        """Non-positive batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DistillationConfig(batch_size=0)

    def test_learning_rate_nonpositive(self) -> None:
        """Non-positive learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            DistillationConfig(learning_rate=-0.001)

    def test_learning_rate_zero(self) -> None:
        """Zero learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            DistillationConfig(learning_rate=0.0)


class TestAugmentationConfig:
    """Tests for AugmentationConfig."""

    def test_default_values(self) -> None:
        """Default AugmentationConfig has all augmentations disabled."""
        cfg = AugmentationConfig()
        assert cfg.horizontal_flip is False
        assert cfg.vertical_flip is False
        assert cfg.rotation is False
        assert cfg.color_jitter is False

    def test_custom_values(self) -> None:
        """Custom augmentation flags are stored correctly."""
        cfg = AugmentationConfig(
            horizontal_flip=True,
            vertical_flip=True,
            rotation=True,
            rotation_degrees=45,
        )
        assert cfg.horizontal_flip is True
        assert cfg.rotation_degrees == 45


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_default_values(self) -> None:
        """Default DataConfig creates successfully."""
        cfg = DataConfig()
        assert cfg.dataset_type == "mvtec_ad"
        assert cfg.image_size == 256

    def test_invalid_dataset_type(self) -> None:
        """Invalid dataset_type raises ValueError."""
        with pytest.raises(ValueError, match="dataset_type must be one of"):
            DataConfig(dataset_type="imagenet")

    def test_valid_dataset_types(self) -> None:
        """All valid dataset types are accepted."""
        for dtype in ("mvtec_ad", "mvtec_loco", "visa"):
            cfg = DataConfig(dataset_type=dtype)
            assert cfg.dataset_type == dtype

    def test_image_size_nonpositive(self) -> None:
        """Non-positive image_size raises ValueError."""
        with pytest.raises(ValueError, match="image_size must be positive"):
            DataConfig(image_size=0)


class TestInferenceConfig:
    """Tests for InferenceConfig validation."""

    def test_default_values(self) -> None:
        """Default InferenceConfig creates successfully."""
        cfg = InferenceConfig()
        assert cfg.normalization == "min_max"
        assert cfg.validation_ratio == 0.1

    def test_invalid_normalization(self) -> None:
        """Invalid normalization method raises ValueError."""
        with pytest.raises(ValueError, match="normalization must be one of"):
            InferenceConfig(normalization="invalid")

    def test_valid_normalization_methods(self) -> None:
        """All valid normalization methods are accepted."""
        for method in ("min_max", "z_score", "rz_score", "none"):
            cfg = InferenceConfig(normalization=method)
            assert cfg.normalization == method

    def test_validation_ratio_out_of_range(self) -> None:
        """Validation ratio outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="validation_ratio must be in"):
            InferenceConfig(validation_ratio=0.0)

        with pytest.raises(ValueError, match="validation_ratio must be in"):
            InferenceConfig(validation_ratio=1.0)

        with pytest.raises(ValueError, match="validation_ratio must be in"):
            InferenceConfig(validation_ratio=-0.1)


class TestConfig:
    """Tests for top-level Config validation."""

    def test_default_values(self) -> None:
        """Default Config creates successfully."""
        cfg = Config()
        assert cfg.device == "cuda"
        assert cfg.output_dir == "runs"
        assert cfg.tensorboard is False

    def test_image_size_cross_validation_mismatch(self) -> None:
        """Mismatched model/data image_size raises ValueError."""
        with pytest.raises(ValueError, match=r"model\.image_size.*must match.*data\.image_size"):
            Config(
                model=ModelConfig(image_size=256),
                data=DataConfig(image_size=128),
            )

    def test_image_size_cross_validation_match(self) -> None:
        """Matching model/data image_size passes validation."""
        cfg = Config(
            model=ModelConfig(image_size=256),
            data=DataConfig(image_size=256),
        )
        assert cfg.model.image_size == cfg.data.image_size
