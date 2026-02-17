"""Shared CLI helpers for argument parsing, config building, and dataset creation.

Provides common utilities used by all CLI subcommands (distill, train, evaluate)
to avoid duplication of argument definitions, override logic, and dataset/dataloader
construction.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from torch.utils.data import DataLoader, Dataset

from mudenet.config.loading import load_config
from mudenet.config.schema import Config, DataConfig
from mudenet.data.datasets import MVTecAD, MVTecLOCO, VisA
from mudenet.data.transforms import (
    get_eval_transform,
    get_mask_transform,
    get_train_transform,
)
from mudenet.data.utils import create_dataloader

logger = logging.getLogger(__name__)

# Mapping from config dataset_type to dataset class
_DATASET_CLASSES: dict[str, type[Dataset]] = {  # type: ignore[type-arg]
    "mvtec_ad": MVTecAD,
    "mvtec_loco": MVTecLOCO,
    "visa": VisA,
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments shared across all subcommands.

    Adds ``--config``, ``--category``, ``--device``, ``--output-dir``,
    and ``--seed`` to the given parser.

    Args:
        parser: Argument parser to add arguments to.
    """
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Dataset category (overrides config, e.g. 'bottle')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (overrides config, e.g. 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and results (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )


def build_overrides(args: argparse.Namespace, seed_target: str) -> dict[str, Any]:
    """Build config override dict from parsed CLI arguments.

    Translates parsed CLI arguments into dot-separated config overrides
    compatible with :func:`mudenet.config.loading.load_config`.

    Args:
        args: Parsed argument namespace containing common CLI fields.
        seed_target: Dot-separated config path for the seed override.
            Use ``"distillation.seed"`` for distill, ``"training.seed"``
            for train and evaluate.

    Returns:
        Dictionary of dot-separated config overrides. Empty values
        (None) are excluded.
    """
    overrides: dict[str, Any] = {}
    if args.category is not None:
        overrides["data.category"] = args.category
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides[seed_target] = args.seed
    return overrides


def load_config_from_subcommand(
    args: argparse.Namespace,
    seed_target: str,
) -> Config:
    """Load and validate config from subcommand arguments.

    Combines YAML config loading with CLI override application.

    Args:
        args: Parsed argument namespace (must contain ``config`` field).
        seed_target: Dot-separated config path for the seed override.

    Returns:
        Fully validated Config instance.
    """
    overrides = build_overrides(args, seed_target)
    return load_config(args.config, overrides=overrides or None)


def get_dataset_class(dataset_type: str) -> type[Dataset]:  # type: ignore[type-arg]
    """Get dataset class for a given dataset type string.

    Args:
        dataset_type: One of ``"mvtec_ad"``, ``"mvtec_loco"``, or ``"visa"``.

    Returns:
        The corresponding dataset class.

    Raises:
        ValueError: If ``dataset_type`` is not recognized.
    """
    if dataset_type not in _DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Expected one of: {sorted(_DATASET_CLASSES.keys())}"
        )
    return _DATASET_CLASSES[dataset_type]


def create_teacher_from_config(config: Config) -> object:
    """Create a TeacherNetwork from config.

    Uses deferred import to preserve fast ``--help`` behavior.

    Args:
        config: Full configuration.

    Returns:
        Freshly initialized TeacherNetwork instance.
    """
    from mudenet.models.teacher import TeacherNetwork

    return TeacherNetwork(
        internal_channels=config.model.internal_channels,
        output_channels=config.model.num_channels,
        num_levels=config.model.num_levels,
        block_depths=config.model.block_depths,
        kernel_sizes=config.model.kernel_sizes,
    )


def create_autoencoder_from_config(config: Config) -> object:
    """Create an Autoencoder from config.

    Uses deferred import to preserve fast ``--help`` behavior.

    Args:
        config: Full configuration.

    Returns:
        Freshly initialized Autoencoder instance.
    """
    from mudenet.models.autoencoder import Autoencoder

    return Autoencoder(
        output_channels=config.model.num_channels,
        latent_dim=config.model.latent_dim,
        num_levels=config.model.num_levels,
    )


def create_dataset(
    config: DataConfig,
    split: str,
) -> Dataset:  # type: ignore[type-arg]
    """Create a dataset instance with appropriate transforms.

    Automatically selects transforms based on the split:
      - ``"train"``: training augmentations, no mask transform
      - ``"test"``: evaluation transforms, with mask transform

    Args:
        config: Data configuration.
        split: Dataset split — ``"train"`` or ``"test"``.

    Returns:
        Configured dataset instance.

    Raises:
        ValueError: If ``split`` is not ``"train"`` or ``"test"``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    dataset_cls = get_dataset_class(config.dataset_type)

    if split == "train":
        transform = get_train_transform(config)
        target_transform = None
    else:
        transform = get_eval_transform(config)
        target_transform = get_mask_transform(config.image_size)

    return dataset_cls(
        data_root=config.data_root,
        category=config.category,
        split=split,
        transform=transform,
        target_transform=target_transform,
    )


def create_dataset_and_loader(
    config: Config,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool | None = None,
    seed: int | None = None,
) -> tuple[Dataset, DataLoader]:  # type: ignore[type-arg]
    """Create a dataset and its DataLoader.

    Convenience function combining :func:`create_dataset` and
    :func:`mudenet.data.utils.create_dataloader`.

    Args:
        config: Full configuration.
        split: Dataset split — ``"train"`` or ``"test"``.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of DataLoader workers.
        shuffle: Whether to shuffle. Defaults to True for train, False for test.
        seed: Random seed for DataLoader shuffling. Defaults to
            ``config.training.seed`` when ``None``.

    Returns:
        Tuple of (dataset, dataloader).
    """
    dataset = create_dataset(config.data, split)

    if shuffle is None:
        shuffle = split == "train"

    loader_seed = seed if seed is not None else config.training.seed

    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=loader_seed,
        pin_memory=(config.device != "cpu"),
    )

    return dataset, loader
