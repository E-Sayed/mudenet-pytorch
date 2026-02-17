"""Configuration loading from YAML files with CLI argument overrides.

Usage:
    config = load_config("configs/default.yaml")
    config = load_config("configs/mvtec_ad.yaml", overrides={"data.category": "bottle"})
    config = load_config_from_args()  # parses sys.argv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from mudenet.config.schema import (
    AugmentationConfig,
    Config,
    DataConfig,
    DistillationConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base dictionary (not modified in place).
        override: Dictionary with values to override.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_model_config(data: dict[str, Any]) -> ModelConfig:
    """Build ModelConfig from a raw dict, converting lists to tuples."""
    data = data.copy()
    if "block_depths" in data and isinstance(data["block_depths"], list):
        data["block_depths"] = tuple(data["block_depths"])
    if "kernel_sizes" in data and isinstance(data["kernel_sizes"], list):
        data["kernel_sizes"] = tuple(data["kernel_sizes"])
    return ModelConfig(**data)


def _build_data_config(data: dict[str, Any]) -> DataConfig:
    """Build DataConfig from a raw dict, handling nested augmentations."""
    data = data.copy()
    aug_data = data.pop("augmentations", {})
    augmentations = AugmentationConfig(**aug_data) if aug_data else AugmentationConfig()
    return DataConfig(augmentations=augmentations, **data)


def _build_config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a Config from a raw dictionary (e.g., parsed YAML).

    Args:
        raw: Dictionary with config sections as keys.

    Returns:
        Fully constructed Config instance.
    """
    known_keys = {
        "model", "training", "distillation", "data", "inference",
        "device", "output_dir", "tensorboard",
    }
    unknown_keys = set(raw.keys()) - known_keys
    if unknown_keys:
        logger.warning("Unknown config keys (ignored): %s", sorted(unknown_keys))

    model = _build_model_config(raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))
    distillation = DistillationConfig(**raw.get("distillation", {}))
    data = _build_data_config(raw.get("data", {}))
    inference = InferenceConfig(**raw.get("inference", {}))

    top_level_keys = {"device", "output_dir", "tensorboard"}
    top_level = {k: v for k, v in raw.items() if k in top_level_keys}

    return Config(
        model=model,
        training=training,
        distillation=distillation,
        data=data,
        inference=inference,
        **top_level,
    )


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from a YAML file with optional overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dictionary of dot-separated key overrides.
            Example: {"data.category": "bottle", "training.num_epochs": 100}

    Returns:
        Fully validated Config instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If config values fail validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if raw_data is None:
        raw: dict[str, Any] = {}
    elif not isinstance(raw_data, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(raw_data).__name__}"
        )
    else:
        raw = raw_data

    logger.info("Loaded config from %s", path)

    # Apply dot-separated overrides
    if overrides:
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        logger.info("Applied %d config overrides", len(overrides))

    return _build_config_from_dict(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser with common CLI options.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="MuDeNet — Multi-patch descriptor network for anomaly detection",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Dataset category (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    return parser


def load_config_from_args(args: list[str] | None = None) -> Config:
    """Load configuration from CLI arguments.

    Parses --config to load the YAML file, then applies any
    additional CLI flags as overrides.

    Args:
        args: Optional argument list (defaults to sys.argv).

    Returns:
        Fully validated Config instance.
    """
    parser = build_arg_parser()
    parsed = parser.parse_args(args)

    overrides: dict[str, Any] = {}
    if parsed.category is not None:
        overrides["data.category"] = parsed.category
    if parsed.device is not None:
        overrides["device"] = parsed.device
    if parsed.output_dir is not None:
        overrides["output_dir"] = parsed.output_dir
    if parsed.seed is not None:
        overrides["training.seed"] = parsed.seed

    return load_config(parsed.config, overrides=overrides or None)
