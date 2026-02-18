"""Stage 2: End-to-end training loop.

Trains S1, A, and S2 jointly with frozen teacher T (Eq. 8, Sec. 4).
The teacher produces target embedding maps, and the three sub-networks
learn to reproduce the structural and logical relationships.

Usage:
    student1, autoencoder, student2 = train_end_to_end(
        teacher, student1, autoencoder, student2, dataloader, config,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mudenet.config.schema import TrainingConfig
from mudenet.models.autoencoder import Autoencoder
from mudenet.models.teacher import TeacherNetwork
from mudenet.training.losses import (
    autoencoder_loss,
    composite_loss,
    logical_loss,
    structural_loss,
)
from mudenet.utils.seed import set_seed

logger = logging.getLogger(__name__)


def train_end_to_end(
    teacher: TeacherNetwork,
    student1: TeacherNetwork,
    autoencoder: Autoencoder,
    student2: TeacherNetwork,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: str = "cuda",
    output_dir: str | Path = "runs",
) -> tuple[TeacherNetwork, Autoencoder, TeacherNetwork]:
    """Train S1, A, S2 jointly with frozen teacher T (Eq. 8).

    The teacher T is frozen (eval mode, no gradients). For each epoch,
    for each batch of nominal images:
      1. teacher_maps = T(images)          — L x (B, C, 128, 128), no grad
      2. student1_maps = S1(images)        — L x (B, C, 128, 128)
      3. autoencoder_maps = A(images)      — L x (B, C, 128, 128)
      4. student2_maps = S2(images)        — L x (B, C, 128, 128)
      5. L1 = structural_loss(teacher_maps, student1_maps)     (Eq. 3)
      6. LA = autoencoder_loss(teacher_maps, autoencoder_maps)  (Eq. 5)
      7. L2 = logical_loss(autoencoder_maps, student2_maps)     (Eq. 7)
      8. total = composite_loss(L1, LA, L2)                     (Eq. 8)
      9. Backprop and update S1, A, S2 jointly

    All three sub-networks share ONE optimizer (Adam).

    Args:
        teacher: Frozen teacher network (pre-trained via distillation).
        student1: Structural student S1 (same architecture as T).
        autoencoder: Autoencoder A (encoder + decoder ensemble).
        student2: Logical student S2 (same architecture as T).
        dataloader: Training data loader (nominal images only).
        config: End-to-end training hyperparameters.
        device: Compute device. Default "cuda".
        output_dir: Directory for saving checkpoints. Default "runs".

    Returns:
        Tuple of (trained_student1, trained_autoencoder, trained_student2).
    """
    if config.mixed_precision:
        raise NotImplementedError(
            "Mixed precision training is not yet supported. "
            "Set mixed_precision=False in TrainingConfig."
        )

    # Set all RNG sources and cuDNN determinism
    set_seed(config.seed, config.deterministic)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Move all models to device
    teacher = teacher.to(device)
    student1 = student1.to(device)
    autoencoder = autoencoder.to(device)
    student2 = student2.to(device)

    # Freeze teacher: eval mode + no gradients
    teacher.eval()
    teacher.requires_grad_(False)

    # Trainable sub-networks in train mode
    student1.train()
    autoencoder.train()
    student2.train()

    # Single Adam optimizer shared across S1, A, S2
    optimizer = torch.optim.Adam(
        list(student1.parameters())
        + list(autoencoder.parameters())
        + list(student2.parameters()),
        lr=config.learning_rate,
    )

    # Optional LR scheduler
    scheduler = None
    if config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=config.lr_min,
        )

    logger.info(
        "Starting end-to-end training: %d epochs, lr=%.1e, lr_schedule=%s, "
        "batch_size=%d, seed=%d",
        config.num_epochs,
        config.learning_rate,
        config.lr_schedule,
        config.batch_size,
        config.seed,
    )

    for epoch in range(config.num_epochs):
        # Ensure train/eval modes at the start of every epoch
        student1.train()
        autoencoder.train()
        student2.train()
        teacher.eval()

        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_la = 0.0
        epoch_l2 = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Training [{epoch + 1}/{config.num_epochs}]",
            leave=False,
        )

        for batch in progress:
            images = batch["image"].to(device)  # (B, 3, 256, 256)

            # Teacher forward pass (frozen, no gradients).
            # Clone outputs so they can participate as targets in loss computation
            # without inference_mode restrictions.
            with torch.inference_mode():
                teacher_maps_raw = teacher(images)  # L x (B, C, 128, 128)
            teacher_maps = [t.clone() for t in teacher_maps_raw]

            # Forward pass through trainable sub-networks
            student1_maps = student1(images)      # L x (B, C, 128, 128)
            autoencoder_maps = autoencoder(images)  # L x (B, C, 128, 128)
            student2_maps = student2(images)      # L x (B, C, 128, 128)

            # Compute sub-losses
            loss_l1 = structural_loss(teacher_maps, student1_maps)     # Eq. 3
            loss_la = autoencoder_loss(teacher_maps, autoencoder_maps)  # Eq. 5
            loss_l2 = logical_loss([m.detach() for m in autoencoder_maps], student2_maps)  # Eq. 7
            total_loss = composite_loss(loss_l1, loss_la, loss_l2)      # Eq. 8

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Cache .item() to avoid redundant CUDA synchronizations
            batch_loss = total_loss.item()
            batch_l1 = loss_l1.item()
            batch_la = loss_la.item()
            batch_l2 = loss_l2.item()

            epoch_loss += batch_loss
            epoch_l1 += batch_l1
            epoch_la += batch_la
            epoch_l2 += batch_l2
            num_batches += 1
            progress.set_postfix(
                loss=f"{batch_loss:.4f}",
                L1=f"{batch_l1:.4f}",
                LA=f"{batch_la:.4f}",
                L2=f"{batch_l2:.4f}",
            )

        if num_batches == 0:
            logger.warning(
                "Epoch %d/%d — dataloader yielded zero batches. "
                "Check dataset and dataloader configuration.",
                epoch + 1,
                config.num_epochs,
            )

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_l1 = epoch_l1 / max(num_batches, 1)
        avg_la = epoch_la / max(num_batches, 1)
        avg_l2 = epoch_l2 / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d — avg loss: %.6f (L1=%.6f, LA=%.6f, L2=%.6f) lr=%.2e",
            epoch + 1,
            config.num_epochs,
            avg_loss,
            avg_l1,
            avg_la,
            avg_l2,
            current_lr,
        )

    # Save checkpoint with all four models and full metadata.
    # teacher_state_dict is included so the evaluate CLI can load all 4 models
    # from a single checkpoint without needing the distillation checkpoint.
    #
    # NOTE: model_config is reconstructed from model internals (e.g.
    # student1.stem.conv.out_channels). This is fragile — if the model
    # architecture changes, these attribute paths break. A cleaner approach
    # would be to pass the full Config into this function and serialize it
    # directly via dataclasses.asdict(). Deferred to a future cleanup
    # (review finding I-6).
    checkpoint_path = output_path / "end_to_end.pt"
    checkpoint: dict[str, object] = {
        "teacher_state_dict": teacher.state_dict(),
        "student1_state_dict": student1.state_dict(),
        "autoencoder_state_dict": autoencoder.state_dict(),
        "student2_state_dict": student2.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "student1": {
                "internal_channels": student1.stem.conv.out_channels,
                "output_channels": student1.projections[0].out_channels,
                "num_levels": student1.num_levels,
                "block_depths": tuple(
                    len(block) for block in student1.blocks
                ),
                "kernel_sizes": tuple(
                    block[0].conv1.kernel_size[0] for block in student1.blocks
                ),
            },
            "autoencoder": {
                "output_channels": autoencoder.decoders[0].layers[0].out_channels,
                "latent_dim": autoencoder.decoders[0].layers[0].in_channels,
                "num_levels": autoencoder.num_levels,
            },
            "student2": {
                "internal_channels": student2.stem.conv.out_channels,
                "output_channels": student2.projections[0].out_channels,
                "num_levels": student2.num_levels,
                "block_depths": tuple(
                    len(block) for block in student2.blocks
                ),
                "kernel_sizes": tuple(
                    block[0].conv1.kernel_size[0] for block in student2.blocks
                ),
            },
        },
        "config": {
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "lr_schedule": config.lr_schedule,
            "lr_min": config.lr_min,
            "batch_size": config.batch_size,
            "seed": config.seed,
        },
        "epoch": config.num_epochs,
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved end-to-end checkpoint to %s", checkpoint_path)

    return student1, autoencoder, student2
