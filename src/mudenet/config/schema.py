"""Configuration dataclasses for MuDeNet.

All hyperparameters, dataset settings, and training options are defined here
as typed dataclasses. See the paper (Sec. 4) for default values.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture hyperparameters.

    Attributes:
        num_channels: Embedding channel dimensionality C. Default 128.
        latent_dim: Autoencoder latent space dimensionality Z. Default 32.
        num_levels: Number of embedding levels L. Default 3.
        internal_channels: Internal channel count for teacher/student networks. Default 64.
        block_depths: Number of residual blocks per level. Default (1, 2, 2).
        kernel_sizes: Kernel size per level. Default (3, 3, 5).
        image_size: Input image spatial size (square). Default 256.
    """

    num_channels: int = 128
    latent_dim: int = 32
    num_levels: int = 3
    internal_channels: int = 64
    block_depths: tuple[int, ...] = (1, 2, 2)
    kernel_sizes: tuple[int, ...] = (3, 3, 5)
    image_size: int = 256

    def __post_init__(self) -> None:
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {self.num_levels}")
        if self.internal_channels <= 0:
            raise ValueError(
                f"internal_channels must be positive, got {self.internal_channels}"
            )
        if len(self.block_depths) != self.num_levels:
            raise ValueError(
                f"block_depths length ({len(self.block_depths)}) must match "
                f"num_levels ({self.num_levels})"
            )
        if len(self.kernel_sizes) != self.num_levels:
            raise ValueError(
                f"kernel_sizes length ({len(self.kernel_sizes)}) must match "
                f"num_levels ({self.num_levels})"
            )
        if any(d <= 0 for d in self.block_depths):
            raise ValueError(
                f"All block_depths must be positive, got {self.block_depths}"
            )
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {self.num_channels}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.image_size != 256:
            raise ValueError(
                f"image_size must be 256 (encoder/decoder architecture is "
                f"hardcoded for this size), got {self.image_size}"
            )


@dataclass
class TrainingConfig:
    """End-to-end training hyperparameters (Stage 2).

    Attributes:
        num_epochs: Number of training epochs. Default 500.
        batch_size: Batch size. Default 8.
        learning_rate: Learning rate for Adam optimizer. Default 1e-3.
        optimizer: Optimizer name. Default "adam".
        seed: Random seed for reproducibility. Default 42.
        deterministic: Enable deterministic operations. Default True.
        mixed_precision: Enable automatic mixed precision. Default False.
        num_workers: DataLoader worker count. Default 4.
    """

    num_epochs: int = 500
    batch_size: int = 8
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    seed: int = 42
    deterministic: bool = True
    mixed_precision: bool = False
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        valid_optimizers = {"adam"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"optimizer must be one of {valid_optimizers}, got '{self.optimizer}'"
            )


@dataclass
class DistillationConfig:
    """Pre-knowledge distillation hyperparameters (Stage 1).

    Attributes:
        backbone: Torchvision model name for feature extraction. Default "wide_resnet50_2".
        num_epochs: Number of distillation epochs. Default 500.
        learning_rate: Learning rate. Default 1e-3.
        batch_size: Batch size. Default 8.
        seed: Random seed for reproducibility. Default 42.
        deterministic: Enable deterministic cuDNN operations. Default True.
    """

    backbone: str = "wide_resnet50_2"
    num_epochs: int = 500
    learning_rate: float = 1e-3
    batch_size: int = 8
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )


@dataclass
class AugmentationConfig:
    """Per-category data augmentation flags.

    See paper Tables A.16-A.18 for per-category settings.

    Attributes:
        horizontal_flip: Enable horizontal flip. Default False.
        vertical_flip: Enable vertical flip. Default False.
        rotation: Enable random rotation. Default False.
        color_jitter: Enable color jitter (brightness, contrast, saturation). Default False.
        rotation_degrees: Max rotation angle in degrees. Default 90.
        brightness: Color jitter brightness factor. Default 0.1.
        contrast: Color jitter contrast factor. Default 0.1.
        saturation: Color jitter saturation factor. Default 0.1.
    """

    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation: bool = False
    color_jitter: bool = False
    rotation_degrees: int = 90
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1


@dataclass
class DataConfig:
    """Dataset configuration.

    Attributes:
        dataset_type: Dataset type identifier ("mvtec_ad", "mvtec_loco", or "visa").
        data_root: Path to dataset root directory.
        category: Category name (e.g. "bottle", "breakfast_box").
        image_size: Input image size (square). Default 256.
        augmentations: Per-category augmentation settings.
    """

    dataset_type: str = "mvtec_ad"
    data_root: str = "data"
    category: str = "bottle"
    image_size: int = 256
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        valid_types = {"mvtec_ad", "mvtec_loco", "visa"}
        if self.dataset_type not in valid_types:
            raise ValueError(
                f"dataset_type must be one of {valid_types}, got '{self.dataset_type}'"
            )
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")


@dataclass
class InferenceConfig:
    """Inference and scoring configuration.

    Attributes:
        normalization: Score normalization method. Default "min_max".
        validation_ratio: Fraction of nominal training data held out for
            computing normalization statistics. Default 0.1.
        smoothing_sigma: Gaussian smoothing sigma applied to the anomaly map
            after upsampling.  Reduces pixel-level noise and improves both
            image-level and region-level metrics.  Set to 0.0 to disable.
            Default 4.0.
    """

    normalization: str = "min_max"
    validation_ratio: float = 0.1
    smoothing_sigma: float = 4.0

    def __post_init__(self) -> None:
        valid_norms = {"min_max", "z_score", "rz_score", "none"}
        if self.normalization not in valid_norms:
            raise ValueError(
                f"normalization must be one of {valid_norms}, got '{self.normalization}'"
            )
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError(
                f"validation_ratio must be in (0, 1), got {self.validation_ratio}"
            )
        if self.smoothing_sigma < 0.0:
            raise ValueError(
                f"smoothing_sigma must be >= 0.0, got {self.smoothing_sigma}"
            )


@dataclass
class Config:
    """Top-level configuration combining all sub-configs.

    Attributes:
        model: Model architecture configuration.
        training: End-to-end training configuration.
        distillation: Pre-knowledge distillation configuration.
        data: Dataset configuration.
        inference: Inference and scoring configuration.
        device: Compute device. Default "cuda".
        output_dir: Directory for saving outputs. Default "runs".
        tensorboard: Enable TensorBoard logging. Default False.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: str = "cuda"
    output_dir: str = "runs"
    tensorboard: bool = False

    def __post_init__(self) -> None:
        if self.model.image_size != self.data.image_size:
            raise ValueError(
                f"model.image_size ({self.model.image_size}) must match "
                f"data.image_size ({self.data.image_size})"
            )
