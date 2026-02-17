"""Tests for the end-to-end training loop (Stage 2).

These tests use small models and tiny synthetic batches — they verify
that the training mechanics work (parameter updates, frozen teacher,
checkpoint saving/loading), not that a full training run produces good results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from mudenet.config.schema import TrainingConfig
from mudenet.models.autoencoder import Autoencoder
from mudenet.models.teacher import TeacherNetwork
from mudenet.training.trainer import train_end_to_end
from tests.conftest import DictDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INTERNAL = 16
_OUTPUT = 32
_LEVELS = 2
_DEPTHS = (1, 1)
_KERNELS = (3, 3)
_LATENT = 8


@pytest.fixture
def small_teacher() -> TeacherNetwork:
    """Small teacher network (frozen during training)."""
    return TeacherNetwork(
        internal_channels=_INTERNAL,
        output_channels=_OUTPUT,
        num_levels=_LEVELS,
        block_depths=_DEPTHS,
        kernel_sizes=_KERNELS,
    )


@pytest.fixture
def small_student1() -> TeacherNetwork:
    """Small structural student S1."""
    return TeacherNetwork(
        internal_channels=_INTERNAL,
        output_channels=_OUTPUT,
        num_levels=_LEVELS,
        block_depths=_DEPTHS,
        kernel_sizes=_KERNELS,
    )


@pytest.fixture
def small_autoencoder() -> Autoencoder:
    """Small autoencoder A."""
    return Autoencoder(
        output_channels=_OUTPUT,
        latent_dim=_LATENT,
        num_levels=_LEVELS,
    )


@pytest.fixture
def small_student2() -> TeacherNetwork:
    """Small logical student S2."""
    return TeacherNetwork(
        internal_channels=_INTERNAL,
        output_channels=_OUTPUT,
        num_levels=_LEVELS,
        block_depths=_DEPTHS,
        kernel_sizes=_KERNELS,
    )


@pytest.fixture
def tiny_dataloader() -> DataLoader:
    """Tiny dataloader with 4 synthetic images."""
    torch.manual_seed(0)
    images = torch.randn(4, 3, 256, 256)
    return DataLoader(DictDataset(images), batch_size=2, shuffle=False)


@pytest.fixture
def training_config() -> TrainingConfig:
    """Minimal training config for testing (1 epoch)."""
    return TrainingConfig(
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests: parameter updates
# ---------------------------------------------------------------------------


class TestParameterUpdates:
    """Verify that trainable models update and teacher stays frozen."""

    def test_student1_parameters_change(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """S1 parameters should change after one epoch of training."""
        initial_params = {
            name: param.clone()
            for name, param in small_student1.named_parameters()
        }

        s1, _, _ = train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        params_changed = any(
            not torch.equal(param, initial_params[name])
            for name, param in s1.named_parameters()
        )
        assert params_changed, "S1 parameters should change after training"

    def test_autoencoder_parameters_change(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Autoencoder parameters should change after one epoch of training."""
        initial_params = {
            name: param.clone()
            for name, param in small_autoencoder.named_parameters()
        }

        _, ae, _ = train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        params_changed = any(
            not torch.equal(param, initial_params[name])
            for name, param in ae.named_parameters()
        )
        assert params_changed, "Autoencoder parameters should change after training"

    def test_student2_parameters_change(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """S2 parameters should change after one epoch of training."""
        initial_params = {
            name: param.clone()
            for name, param in small_student2.named_parameters()
        }

        _, _, s2 = train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        params_changed = any(
            not torch.equal(param, initial_params[name])
            for name, param in s2.named_parameters()
        )
        assert params_changed, "S2 parameters should change after training"

    def test_teacher_parameters_unchanged(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Teacher parameters must NOT change during end-to-end training."""
        initial_params = {
            name: param.clone()
            for name, param in small_teacher.named_parameters()
        }

        train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        for name, param in small_teacher.named_parameters():
            assert torch.equal(param, initial_params[name]), (
                f"Teacher parameter '{name}' should not change during training"
            )


# ---------------------------------------------------------------------------
# Tests: checkpoint saving
# ---------------------------------------------------------------------------


class TestCheckpointSaving:
    """Verify checkpoint is saved with all required fields."""

    def test_checkpoint_file_exists(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """A checkpoint file is saved after training."""
        train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint_path = tmp_path / "end_to_end.pt"
        assert checkpoint_path.exists(), "Checkpoint file should be saved"

    def test_checkpoint_has_required_keys(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Checkpoint contains all expected keys."""
        train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint = torch.load(tmp_path / "end_to_end.pt", weights_only=False)
        expected_keys = {
            "student1_state_dict",
            "autoencoder_state_dict",
            "student2_state_dict",
            "optimizer_state_dict",
            "model_config",
            "config",
            "epoch",
        }
        assert expected_keys.issubset(checkpoint.keys()), (
            f"Missing keys: {expected_keys - checkpoint.keys()}"
        )

    def test_checkpoint_epoch_value(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Checkpoint records the correct epoch count."""
        train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint = torch.load(tmp_path / "end_to_end.pt", weights_only=False)
        assert checkpoint["epoch"] == training_config.num_epochs

    def test_checkpoint_model_config_structure(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """model_config contains architecture info for all three sub-networks."""
        train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint = torch.load(tmp_path / "end_to_end.pt", weights_only=False)
        model_config = checkpoint["model_config"]

        assert "student1" in model_config
        assert "autoencoder" in model_config
        assert "student2" in model_config

        # Verify student config values match
        assert model_config["student1"]["internal_channels"] == _INTERNAL
        assert model_config["student1"]["output_channels"] == _OUTPUT
        assert model_config["student1"]["num_levels"] == _LEVELS

        # Verify autoencoder config values match
        assert model_config["autoencoder"]["output_channels"] == _OUTPUT
        assert model_config["autoencoder"]["latent_dim"] == _LATENT
        assert model_config["autoencoder"]["num_levels"] == _LEVELS


# ---------------------------------------------------------------------------
# Tests: checkpoint loading
# ---------------------------------------------------------------------------


class TestCheckpointLoading:
    """Verify checkpoints can be loaded into fresh models."""

    def test_checkpoint_loadable_into_fresh_models(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Saved state dicts can be loaded into fresh model instances."""
        train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint = torch.load(tmp_path / "end_to_end.pt", weights_only=False)

        # Create fresh models with same architecture
        fresh_s1 = TeacherNetwork(
            internal_channels=_INTERNAL,
            output_channels=_OUTPUT,
            num_levels=_LEVELS,
            block_depths=_DEPTHS,
            kernel_sizes=_KERNELS,
        )
        fresh_ae = Autoencoder(
            output_channels=_OUTPUT,
            latent_dim=_LATENT,
            num_levels=_LEVELS,
        )
        fresh_s2 = TeacherNetwork(
            internal_channels=_INTERNAL,
            output_channels=_OUTPUT,
            num_levels=_LEVELS,
            block_depths=_DEPTHS,
            kernel_sizes=_KERNELS,
        )

        # Load state dicts — should not raise
        fresh_s1.load_state_dict(checkpoint["student1_state_dict"])
        fresh_ae.load_state_dict(checkpoint["autoencoder_state_dict"])
        fresh_s2.load_state_dict(checkpoint["student2_state_dict"])

    def test_loaded_models_produce_same_output(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Fresh models loaded from checkpoint produce identical outputs."""
        s1, ae, s2 = train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint = torch.load(tmp_path / "end_to_end.pt", weights_only=False)

        fresh_s1 = TeacherNetwork(
            internal_channels=_INTERNAL,
            output_channels=_OUTPUT,
            num_levels=_LEVELS,
            block_depths=_DEPTHS,
            kernel_sizes=_KERNELS,
        )
        fresh_ae = Autoencoder(
            output_channels=_OUTPUT,
            latent_dim=_LATENT,
            num_levels=_LEVELS,
        )
        fresh_s2 = TeacherNetwork(
            internal_channels=_INTERNAL,
            output_channels=_OUTPUT,
            num_levels=_LEVELS,
            block_depths=_DEPTHS,
            kernel_sizes=_KERNELS,
        )

        fresh_s1.load_state_dict(checkpoint["student1_state_dict"])
        fresh_ae.load_state_dict(checkpoint["autoencoder_state_dict"])
        fresh_s2.load_state_dict(checkpoint["student2_state_dict"])

        # Compare outputs in eval mode
        x = torch.randn(1, 3, 256, 256)
        s1.eval()
        ae.eval()
        s2.eval()
        fresh_s1.eval()
        fresh_ae.eval()
        fresh_s2.eval()

        with torch.inference_mode():
            for a, b in zip(s1(x), fresh_s1(x), strict=True):
                assert torch.equal(a, b), "S1 outputs should match after loading"
            for a, b in zip(ae(x), fresh_ae(x), strict=True):
                assert torch.equal(a, b), "Autoencoder outputs should match after loading"
            for a, b in zip(s2(x), fresh_s2(x), strict=True):
                assert torch.equal(a, b), "S2 outputs should match after loading"


# ---------------------------------------------------------------------------
# Tests: return values
# ---------------------------------------------------------------------------


class TestReturnValues:
    """Verify the function returns the trained models."""

    def test_returns_tuple_of_three(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Returns a tuple of (S1, A, S2)."""
        result = train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_same_objects(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Returns the same model instances that were passed in."""
        s1, ae, s2 = train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        assert s1 is small_student1
        assert ae is small_autoencoder
        assert s2 is small_student2


# ---------------------------------------------------------------------------
# Tests: loss behaviour
# ---------------------------------------------------------------------------


class TestLossBehaviour:
    """Verify loss decreases during training."""

    def test_loss_decreases(self, tmp_path: Path) -> None:
        """With 2 epochs, loss at epoch 2 should be <= loss at epoch 1.

        Uses a deterministic setup with fixed seed to make this reliable.
        """
        torch.manual_seed(42)

        teacher = TeacherNetwork(
            internal_channels=_INTERNAL,
            output_channels=_OUTPUT,
            num_levels=_LEVELS,
            block_depths=_DEPTHS,
            kernel_sizes=_KERNELS,
        )
        student1 = TeacherNetwork(
            internal_channels=_INTERNAL,
            output_channels=_OUTPUT,
            num_levels=_LEVELS,
            block_depths=_DEPTHS,
            kernel_sizes=_KERNELS,
        )
        autoencoder = Autoencoder(
            output_channels=_OUTPUT,
            latent_dim=_LATENT,
            num_levels=_LEVELS,
        )
        student2 = TeacherNetwork(
            internal_channels=_INTERNAL,
            output_channels=_OUTPUT,
            num_levels=_LEVELS,
            block_depths=_DEPTHS,
            kernel_sizes=_KERNELS,
        )

        images = torch.randn(2, 3, 256, 256)
        loader = DataLoader(DictDataset(images), batch_size=2, shuffle=False)

        config = TrainingConfig(num_epochs=2, batch_size=2, learning_rate=1e-3, seed=42)

        # Capture loss values from logging
        losses: list[float] = []
        trainer_logger = logging.getLogger("mudenet.training.trainer")
        original_level = trainer_logger.level

        class _LossCapture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                msg = record.getMessage()
                if "avg loss:" in msg:
                    loss_str = msg.split("avg loss: ")[1].split(" ")[0]
                    losses.append(float(loss_str))

        handler = _LossCapture()
        trainer_logger.addHandler(handler)
        trainer_logger.setLevel(logging.INFO)

        try:
            train_end_to_end(
                teacher=teacher,
                student1=student1,
                autoencoder=autoencoder,
                student2=student2,
                dataloader=loader,
                config=config,
                device="cpu",
                output_dir=str(tmp_path),
            )
        finally:
            trainer_logger.removeHandler(handler)
            trainer_logger.setLevel(original_level)

        assert len(losses) == 2, f"Expected 2 loss values, got {len(losses)}"
        assert losses[1] <= losses[0], (
            f"Loss should decrease: epoch 1={losses[0]:.6f}, epoch 2={losses[1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Tests: checkpoint model_config reconstruction
# ---------------------------------------------------------------------------


class TestCheckpointReconstruction:
    """Verify models can be reconstructed using only checkpoint model_config."""

    def test_reconstruct_from_model_config(
        self,
        small_teacher: TeacherNetwork,
        small_student1: TeacherNetwork,
        small_autoencoder: Autoencoder,
        small_student2: TeacherNetwork,
        tiny_dataloader: DataLoader,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Models reconstructed from model_config load state dicts successfully."""
        train_end_to_end(
            teacher=small_teacher,
            student1=small_student1,
            autoencoder=small_autoencoder,
            student2=small_student2,
            dataloader=tiny_dataloader,
            config=training_config,
            device="cpu",
            output_dir=str(tmp_path),
        )

        checkpoint = torch.load(tmp_path / "end_to_end.pt", weights_only=False)
        mc = checkpoint["model_config"]

        # Reconstruct S1 using ONLY checkpoint model_config
        s1_cfg = mc["student1"]
        fresh_s1 = TeacherNetwork(
            internal_channels=s1_cfg["internal_channels"],
            output_channels=s1_cfg["output_channels"],
            num_levels=s1_cfg["num_levels"],
            block_depths=s1_cfg["block_depths"],
            kernel_sizes=s1_cfg["kernel_sizes"],
        )
        fresh_s1.load_state_dict(checkpoint["student1_state_dict"])

        # Reconstruct A using ONLY checkpoint model_config
        ae_cfg = mc["autoencoder"]
        fresh_ae = Autoencoder(
            output_channels=ae_cfg["output_channels"],
            latent_dim=ae_cfg["latent_dim"],
            num_levels=ae_cfg["num_levels"],
        )
        fresh_ae.load_state_dict(checkpoint["autoencoder_state_dict"])

        # Reconstruct S2 using ONLY checkpoint model_config
        s2_cfg = mc["student2"]
        fresh_s2 = TeacherNetwork(
            internal_channels=s2_cfg["internal_channels"],
            output_channels=s2_cfg["output_channels"],
            num_levels=s2_cfg["num_levels"],
            block_depths=s2_cfg["block_depths"],
            kernel_sizes=s2_cfg["kernel_sizes"],
        )
        fresh_s2.load_state_dict(checkpoint["student2_state_dict"])
