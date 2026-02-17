"""Tests for CLI entry points (Step 8).

Tests argument parsing, subcommand registration, and dispatcher wiring.
Does NOT test actual training/evaluation execution — that requires GPU
and real datasets. Tests only parsing, wiring, and help output.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from mudenet.cli.__main__ import main
from mudenet.cli.distill import add_distill_parser, run_distill
from mudenet.cli.evaluate import add_evaluate_parser, run_evaluate
from mudenet.cli.train import add_train_parser, run_train
from mudenet.cli.utils import (
    add_common_args,
    build_overrides,
    create_dataset,
    create_dataset_and_loader,
    get_dataset_class,
    load_config_from_subcommand,
)
from mudenet.config.schema import Config, DataConfig

# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------


class TestMainDispatcher:
    """Test the top-level CLI dispatcher."""

    def test_no_args_exits_cleanly(self) -> None:
        """main() with no arguments prints help and exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_distill_help_exits_cleanly(self) -> None:
        """'distill --help' exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["distill", "--help"])
        assert exc_info.value.code == 0

    def test_train_help_exits_cleanly(self) -> None:
        """'train --help' exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--help"])
        assert exc_info.value.code == 0

    def test_evaluate_help_exits_cleanly(self) -> None:
        """'evaluate --help' exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["evaluate", "--help"])
        assert exc_info.value.code == 0

    def test_invalid_subcommand_exits_with_error(self) -> None:
        """Invalid subcommand causes argparse to exit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent"])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Subparser registration tests
# ---------------------------------------------------------------------------


def _make_subparsers() -> argparse._SubParsersAction:  # type: ignore[type-arg]
    """Create a fresh parser with subparsers for testing."""
    parser = argparse.ArgumentParser(prog="mudenet")
    return parser.add_subparsers(dest="subcommand")


class TestSubparserRegistration:
    """Test that each subcommand registers its parser correctly."""

    def test_distill_parser_registers(self) -> None:
        """add_distill_parser creates a 'distill' subcommand."""
        subparsers = _make_subparsers()
        add_distill_parser(subparsers)
        assert "distill" in subparsers.choices

    def test_train_parser_registers(self) -> None:
        """add_train_parser creates a 'train' subcommand."""
        subparsers = _make_subparsers()
        add_train_parser(subparsers)
        assert "train" in subparsers.choices

    def test_evaluate_parser_registers(self) -> None:
        """add_evaluate_parser creates an 'evaluate' subcommand."""
        subparsers = _make_subparsers()
        add_evaluate_parser(subparsers)
        assert "evaluate" in subparsers.choices


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


class TestDistillArgParsing:
    """Test argument parsing for the 'distill' subcommand."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")
        add_distill_parser(subparsers)
        return parser.parse_args(["distill", *argv])

    def test_default_config(self) -> None:
        """Default --config is configs/default.yaml."""
        args = self._parse([])
        assert args.config == "configs/default.yaml"

    def test_custom_config(self) -> None:
        """--config overrides the default."""
        args = self._parse(["--config", "configs/mvtec_ad.yaml"])
        assert args.config == "configs/mvtec_ad.yaml"

    def test_category_optional(self) -> None:
        """--category is optional for distill (defaults to None)."""
        args = self._parse([])
        assert args.category is None

    def test_all_args_parsed(self) -> None:
        """All common args are parsed correctly."""
        args = self._parse([
            "--config", "configs/mvtec_ad.yaml",
            "--category", "bottle",
            "--device", "cpu",
            "--output-dir", "output",
            "--seed", "123",
        ])
        assert args.config == "configs/mvtec_ad.yaml"
        assert args.category == "bottle"
        assert args.device == "cpu"
        assert args.output_dir == "output"
        assert args.seed == 123

    def test_func_set_to_run_distill(self) -> None:
        """The 'func' default is set to run_distill."""
        args = self._parse([])
        assert args.func is run_distill


class TestTrainArgParsing:
    """Test argument parsing for the 'train' subcommand."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")
        add_train_parser(subparsers)
        return parser.parse_args(["train", *argv])

    def test_checkpoint_required(self) -> None:
        """--checkpoint is required for train."""
        with pytest.raises(SystemExit) as exc_info:
            self._parse([])
        assert exc_info.value.code == 2

    def test_checkpoint_parsed(self) -> None:
        """--checkpoint value is stored."""
        args = self._parse(["--checkpoint", "runs/teacher_distilled.pt"])
        assert args.checkpoint == "runs/teacher_distilled.pt"

    def test_all_args_parsed(self) -> None:
        """All args for train are parsed correctly."""
        args = self._parse([
            "--config", "configs/mvtec_ad.yaml",
            "--category", "bottle",
            "--checkpoint", "runs/teacher_distilled.pt",
            "--device", "cuda",
            "--output-dir", "runs/bottle",
            "--seed", "99",
        ])
        assert args.config == "configs/mvtec_ad.yaml"
        assert args.category == "bottle"
        assert args.checkpoint == "runs/teacher_distilled.pt"
        assert args.device == "cuda"
        assert args.output_dir == "runs/bottle"
        assert args.seed == 99

    def test_func_set_to_run_train(self) -> None:
        """The 'func' default is set to run_train."""
        args = self._parse(["--checkpoint", "x.pt"])
        assert args.func is run_train


class TestEvaluateArgParsing:
    """Test argument parsing for the 'evaluate' subcommand."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")
        add_evaluate_parser(subparsers)
        return parser.parse_args(["evaluate", *argv])

    def test_checkpoint_required(self) -> None:
        """--checkpoint is required for evaluate."""
        with pytest.raises(SystemExit) as exc_info:
            self._parse([])
        assert exc_info.value.code == 2

    def test_checkpoint_parsed(self) -> None:
        """--checkpoint value is stored."""
        args = self._parse(["--checkpoint", "runs/end_to_end.pt"])
        assert args.checkpoint == "runs/end_to_end.pt"

    def test_all_args_parsed(self) -> None:
        """All args for evaluate are parsed correctly."""
        args = self._parse([
            "--config", "configs/mvtec_ad.yaml",
            "--category", "bottle",
            "--checkpoint", "runs/end_to_end.pt",
            "--device", "cpu",
            "--output-dir", "results",
            "--seed", "0",
        ])
        assert args.config == "configs/mvtec_ad.yaml"
        assert args.category == "bottle"
        assert args.checkpoint == "runs/end_to_end.pt"
        assert args.device == "cpu"
        assert args.output_dir == "results"
        assert args.seed == 0

    def test_func_set_to_run_evaluate(self) -> None:
        """The 'func' default is set to run_evaluate."""
        args = self._parse(["--checkpoint", "x.pt"])
        assert args.func is run_evaluate


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestCommonArgs:
    """Test the add_common_args helper."""

    def test_adds_all_common_args(self) -> None:
        """add_common_args adds --config, --category, --device, --output-dir, --seed."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([
            "--config", "c.yaml",
            "--category", "bottle",
            "--device", "cpu",
            "--output-dir", "out",
            "--seed", "7",
        ])
        assert args.config == "c.yaml"
        assert args.category == "bottle"
        assert args.device == "cpu"
        assert args.output_dir == "out"
        assert args.seed == 7

    def test_defaults_are_correct(self) -> None:
        """Common args default to expected values."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        assert args.config == "configs/default.yaml"
        assert args.category is None
        assert args.device is None
        assert args.output_dir is None
        assert args.seed is None


class TestBuildOverrides:
    """Test the build_overrides helper."""

    def test_empty_when_all_none(self) -> None:
        """No overrides when all args are None."""
        args = argparse.Namespace(
            category=None, device=None, output_dir=None, seed=None,
        )
        overrides = build_overrides(args, seed_target="training.seed")
        assert overrides == {}

    def test_category_override(self) -> None:
        """--category maps to data.category override."""
        args = argparse.Namespace(
            category="bottle", device=None, output_dir=None, seed=None,
        )
        overrides = build_overrides(args, seed_target="training.seed")
        assert overrides == {"data.category": "bottle"}

    def test_seed_target_distillation(self) -> None:
        """--seed maps to distillation.seed for distill command."""
        args = argparse.Namespace(
            category=None, device=None, output_dir=None, seed=42,
        )
        overrides = build_overrides(args, seed_target="distillation.seed")
        assert overrides == {"distillation.seed": 42}

    def test_seed_target_training(self) -> None:
        """--seed maps to training.seed for train command."""
        args = argparse.Namespace(
            category=None, device=None, output_dir=None, seed=42,
        )
        overrides = build_overrides(args, seed_target="training.seed")
        assert overrides == {"training.seed": 42}

    def test_all_overrides(self) -> None:
        """All non-None args produce overrides."""
        args = argparse.Namespace(
            category="screw", device="cpu", output_dir="/tmp/out", seed=7,
        )
        overrides = build_overrides(args, seed_target="training.seed")
        assert overrides == {
            "data.category": "screw",
            "device": "cpu",
            "output_dir": "/tmp/out",
            "training.seed": 7,
        }


class TestGetDatasetClass:
    """Test the get_dataset_class helper."""

    def test_mvtec_ad(self) -> None:
        """'mvtec_ad' returns MVTecAD class."""
        from mudenet.data.datasets import MVTecAD
        assert get_dataset_class("mvtec_ad") is MVTecAD

    def test_mvtec_loco(self) -> None:
        """'mvtec_loco' returns MVTecLOCO class."""
        from mudenet.data.datasets import MVTecLOCO
        assert get_dataset_class("mvtec_loco") is MVTecLOCO

    def test_visa(self) -> None:
        """'visa' returns VisA class."""
        from mudenet.data.datasets import VisA
        assert get_dataset_class("visa") is VisA

    def test_unknown_raises(self) -> None:
        """Unknown dataset type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset_type"):
            get_dataset_class("unknown")


# ---------------------------------------------------------------------------
# load_config_from_subcommand tests
# ---------------------------------------------------------------------------


class TestLoadConfigFromSubcommand:
    """Test load_config_from_subcommand."""

    @patch("mudenet.cli.utils.load_config")
    def test_calls_load_config_with_overrides(
        self, mock_load: MagicMock
    ) -> None:
        """Passes CLI overrides to load_config correctly."""
        mock_load.return_value = Config()
        args = argparse.Namespace(
            config="configs/default.yaml",
            category="bottle",
            device="cpu",
            output_dir=None,
            seed=123,
        )
        load_config_from_subcommand(args, seed_target="training.seed")
        mock_load.assert_called_once_with(
            "configs/default.yaml",
            overrides={
                "data.category": "bottle",
                "device": "cpu",
                "training.seed": 123,
            },
        )

    @patch("mudenet.cli.utils.load_config")
    def test_no_overrides_passes_none(self, mock_load: MagicMock) -> None:
        """No CLI overrides passes overrides=None."""
        mock_load.return_value = Config()
        args = argparse.Namespace(
            config="configs/default.yaml",
            category=None,
            device=None,
            output_dir=None,
            seed=None,
        )
        load_config_from_subcommand(args, seed_target="training.seed")
        mock_load.assert_called_once_with(
            "configs/default.yaml",
            overrides=None,
        )

    @patch("mudenet.cli.utils.load_config")
    def test_distillation_seed_target(self, mock_load: MagicMock) -> None:
        """seed_target='distillation.seed' routes --seed correctly."""
        mock_load.return_value = Config()
        args = argparse.Namespace(
            config="configs/default.yaml",
            category=None,
            device=None,
            output_dir=None,
            seed=99,
        )
        load_config_from_subcommand(args, seed_target="distillation.seed")
        mock_load.assert_called_once_with(
            "configs/default.yaml",
            overrides={"distillation.seed": 99},
        )


# ---------------------------------------------------------------------------
# create_dataset tests
# ---------------------------------------------------------------------------


class TestCreateDataset:
    """Test create_dataset helper."""

    @patch("mudenet.cli.utils.get_dataset_class")
    @patch("mudenet.cli.utils.get_train_transform")
    def test_train_split_uses_train_transform(
        self, mock_train_tx: MagicMock, mock_cls: MagicMock
    ) -> None:
        """split='train' applies training transforms, no mask transform."""
        mock_dataset = MagicMock()
        mock_cls.return_value = mock_dataset
        config = DataConfig()

        create_dataset(config, split="train")

        mock_train_tx.assert_called_once_with(config)
        mock_dataset.assert_called_once()
        call_kwargs = mock_dataset.call_args
        assert call_kwargs.kwargs.get("target_transform") is None

    @patch("mudenet.cli.utils.get_dataset_class")
    @patch("mudenet.cli.utils.get_eval_transform")
    @patch("mudenet.cli.utils.get_mask_transform")
    def test_test_split_uses_eval_transform_and_mask(
        self,
        mock_mask_tx: MagicMock,
        mock_eval_tx: MagicMock,
        mock_cls: MagicMock,
    ) -> None:
        """split='test' applies eval transforms with mask transform."""
        mock_dataset = MagicMock()
        mock_cls.return_value = mock_dataset
        config = DataConfig()

        create_dataset(config, split="test")

        mock_eval_tx.assert_called_once_with(config)
        mock_mask_tx.assert_called_once_with(config.image_size)
        mock_dataset.assert_called_once()
        call_kwargs = mock_dataset.call_args
        assert call_kwargs.kwargs.get("target_transform") is not None

    def test_invalid_split_raises(self) -> None:
        """Invalid split raises ValueError."""
        config = DataConfig()
        with pytest.raises(ValueError, match="split must be"):
            create_dataset(config, split="validation")


# ---------------------------------------------------------------------------
# create_dataset_and_loader tests
# ---------------------------------------------------------------------------


class TestCreateDatasetAndLoader:
    """Test create_dataset_and_loader helper."""

    @patch("mudenet.cli.utils.create_dataloader")
    @patch("mudenet.cli.utils.create_dataset")
    def test_train_defaults_shuffle_true(
        self, mock_ds: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Train split defaults to shuffle=True."""
        mock_ds.return_value = MagicMock()
        mock_loader.return_value = MagicMock()
        config = Config()

        create_dataset_and_loader(
            config, split="train", batch_size=8, num_workers=0,
        )

        call_kwargs = mock_loader.call_args
        assert call_kwargs.kwargs.get("shuffle") is True

    @patch("mudenet.cli.utils.create_dataloader")
    @patch("mudenet.cli.utils.create_dataset")
    def test_test_defaults_shuffle_false(
        self, mock_ds: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Test split defaults to shuffle=False."""
        mock_ds.return_value = MagicMock()
        mock_loader.return_value = MagicMock()
        config = Config()

        create_dataset_and_loader(
            config, split="test", batch_size=8, num_workers=0,
        )

        call_kwargs = mock_loader.call_args
        assert call_kwargs.kwargs.get("shuffle") is False

    @patch("mudenet.cli.utils.create_dataloader")
    @patch("mudenet.cli.utils.create_dataset")
    def test_custom_seed_propagated(
        self, mock_ds: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Custom seed parameter is passed to create_dataloader."""
        mock_ds.return_value = MagicMock()
        mock_loader.return_value = MagicMock()
        config = Config()

        create_dataset_and_loader(
            config, split="train", batch_size=8, num_workers=0, seed=999,
        )

        call_kwargs = mock_loader.call_args
        assert call_kwargs.kwargs.get("seed") == 999

    @patch("mudenet.cli.utils.create_dataloader")
    @patch("mudenet.cli.utils.create_dataset")
    def test_default_seed_from_config(
        self, mock_ds: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Default seed comes from config.training.seed."""
        mock_ds.return_value = MagicMock()
        mock_loader.return_value = MagicMock()
        config = Config()
        config.training.seed = 777

        create_dataset_and_loader(
            config, split="train", batch_size=8, num_workers=0,
        )

        call_kwargs = mock_loader.call_args
        assert call_kwargs.kwargs.get("seed") == 777
