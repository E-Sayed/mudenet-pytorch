"""Tests for configuration loading from YAML files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from mudenet.config.loading import (
    _build_config_from_dict,
    _build_data_config,
    _build_model_config,
    load_config,
)
from mudenet.config.schema import Config


@pytest.fixture
def tmp_yaml(tmp_path: Path) -> Path:
    """Create a minimal valid YAML config file."""
    config = {
        "model": {
            "num_channels": 128,
            "image_size": 256,
        },
        "data": {
            "category": "bottle",
            "image_size": 256,
        },
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def full_yaml(tmp_path: Path) -> Path:
    """Create a full valid YAML config file with all sections."""
    config = {
        "model": {
            "num_channels": 128,
            "latent_dim": 32,
            "num_levels": 3,
            "internal_channels": 64,
            "block_depths": [1, 2, 2],
            "kernel_sizes": [3, 3, 5],
            "image_size": 256,
        },
        "training": {
            "num_epochs": 100,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
        "distillation": {
            "num_epochs": 200,
            "learning_rate": 0.001,
        },
        "data": {
            "dataset_type": "mvtec_ad",
            "data_root": "data",
            "category": "bottle",
            "image_size": 256,
            "augmentations": {
                "horizontal_flip": True,
                "vertical_flip": False,
            },
        },
        "inference": {
            "normalization": "min_max",
            "validation_ratio": 0.1,
        },
        "device": "cpu",
        "output_dir": "test_runs",
        "tensorboard": True,
    }
    path = tmp_path / "full_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return path


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_minimal(self, tmp_yaml: Path) -> None:
        """Loading a minimal YAML config works with defaults."""
        cfg = load_config(tmp_yaml)
        assert isinstance(cfg, Config)
        assert cfg.data.category == "bottle"

    def test_load_full(self, full_yaml: Path) -> None:
        """Loading a full YAML config populates all fields."""
        cfg = load_config(full_yaml)
        assert cfg.training.num_epochs == 100
        assert cfg.training.batch_size == 4
        assert cfg.device == "cpu"
        assert cfg.output_dir == "test_runs"
        assert cfg.tensorboard is True
        assert cfg.data.augmentations.horizontal_flip is True
        assert cfg.data.augmentations.vertical_flip is False

    def test_load_with_overrides(self, tmp_yaml: Path) -> None:
        """Dot-separated overrides are applied correctly."""
        cfg = load_config(tmp_yaml, overrides={"data.category": "cable"})
        assert cfg.data.category == "cable"

    def test_load_nested_override(self, tmp_yaml: Path) -> None:
        """Nested dot override creates intermediate dicts."""
        cfg = load_config(tmp_yaml, overrides={"training.num_epochs": 100})
        assert cfg.training.num_epochs == 100

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML file produces default config."""
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        cfg = load_config(path)
        assert isinstance(cfg, Config)

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        """YAML file containing a non-dict raises ValueError."""
        path = tmp_path / "list.yaml"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(path)

    def test_scalar_yaml_raises(self, tmp_path: Path) -> None:
        """YAML file containing a scalar raises ValueError."""
        path = tmp_path / "scalar.yaml"
        path.write_text("42", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(path)

    def test_encoding_utf8(self, tmp_path: Path) -> None:
        """Config file with UTF-8 characters loads correctly."""
        config = {"data": {"category": "café", "image_size": 256}}
        path = tmp_path / "utf8.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        cfg = load_config(path)
        assert cfg.data.category == "café"


class TestListToTupleConversion:
    """Tests for YAML list to Python tuple conversion in loading."""

    def test_block_depths_list_to_tuple(self) -> None:
        """YAML lists for block_depths are converted to tuples."""
        data: dict[str, Any] = {
            "block_depths": [1, 2, 2],
            "kernel_sizes": [3, 3, 5],
        }
        cfg = _build_model_config(data)
        assert isinstance(cfg.block_depths, tuple)
        assert cfg.block_depths == (1, 2, 2)

    def test_kernel_sizes_list_to_tuple(self) -> None:
        """YAML lists for kernel_sizes are converted to tuples."""
        data: dict[str, Any] = {
            "block_depths": [1, 2, 2],
            "kernel_sizes": [3, 3, 5],
        }
        cfg = _build_model_config(data)
        assert isinstance(cfg.kernel_sizes, tuple)
        assert cfg.kernel_sizes == (3, 3, 5)

    def test_tuples_preserved(self) -> None:
        """Already-tuple values are not broken."""
        data: dict[str, Any] = {
            "block_depths": (1, 2, 2),
            "kernel_sizes": (3, 3, 5),
        }
        cfg = _build_model_config(data)
        assert cfg.block_depths == (1, 2, 2)


class TestInputDictNotMutated:
    """Tests that builder functions don't mutate their input dicts."""

    def test_build_model_config_no_mutation(self) -> None:
        """_build_model_config does not modify the input dict."""
        data: dict[str, Any] = {
            "block_depths": [1, 2, 2],
            "kernel_sizes": [3, 3, 5],
        }
        original = data.copy()
        _build_model_config(data)
        assert data == original, "Input dict was mutated by _build_model_config"

    def test_build_data_config_no_mutation(self) -> None:
        """_build_data_config does not modify the input dict."""
        data: dict[str, Any] = {
            "category": "bottle",
            "augmentations": {"horizontal_flip": True},
        }
        original_keys = set(data.keys())
        _build_data_config(data)
        assert set(data.keys()) == original_keys, "Input dict was mutated by _build_data_config"


class TestUnknownKeysWarning:
    """Tests that unknown top-level config keys produce a warning."""

    def test_unknown_keys_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown top-level keys produce a warning log."""
        raw: dict[str, Any] = {
            "taining": {"num_epochs": 100},  # typo for "training"
        }
        with caplog.at_level(logging.WARNING):
            _build_config_from_dict(raw)

        assert any("Unknown config keys" in record.message for record in caplog.records)
        assert any("taining" in record.message for record in caplog.records)

    def test_known_keys_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Known top-level keys do not produce a warning."""
        raw: dict[str, Any] = {
            "model": {},
            "training": {},
            "device": "cpu",
        }
        with caplog.at_level(logging.WARNING):
            _build_config_from_dict(raw)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 0
