"""Stage 1: Pre-knowledge distillation training loop.

Trains the teacher network T by distilling from a pretrained WideResNet50-2.
The feature extractor produces target embeddings E, and the teacher learns to
reproduce them at all L levels (Eq. 16, Sec. 3.3).

Usage:
    teacher = train_distillation(teacher, feature_extractor, dataloader, config)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from mudenet.config.schema import DistillationConfig
from mudenet.models.feature_extractor import FeatureExtractor
from mudenet.models.teacher import TeacherNetwork
from mudenet.training.losses import distillation_loss
from mudenet.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _upsample_target(
    target: Tensor,
    spatial_size: tuple[int, int],
) -> Tensor:
    """Upsample distillation target to match teacher output resolution.

    Resolves A-009: feature extractor outputs at 64x64, teacher at 128x128.
    Bilinear interpolation is used for smooth upsampling.

    E5.7 tested nearest-neighbor (Kronecker-product equivalent) but it
    regressed PRO by -1.0pp and I-AUROC by -0.6pp. Bilinear is kept.

    Args:
        target: Distillation target (B, C, H1, W1) — e.g. (B, 128, 64, 64).
        spatial_size: Target spatial dimensions (H, W) — e.g. (128, 128).

    Returns:
        Upsampled target (B, C, H, W).
    """
    if target.shape[2:] == spatial_size:
        return target
    return F.interpolate(
        target,
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )


def train_distillation(
    teacher: TeacherNetwork,
    feature_extractor: FeatureExtractor,
    dataloader: DataLoader,
    config: DistillationConfig,
    device: str = "cuda",
    output_dir: str = "runs",
) -> TeacherNetwork:
    """Train teacher via knowledge distillation from WideResNet50-2 (Eq. 16).

    The feature extractor is frozen (eval mode, no gradients). For each
    training image, we:
      1. Extract E = fe(images) — shape (B, C, 64, 64)
      2. Upsample E to 128x128 to match teacher output resolution (A-009)
      3. Compute teacher_maps = teacher(images) — list of L x (B, C, 128, 128)
      4. Loss = mean over L levels of ||E_upsampled - teacher_map_l||^2_F
         (sum over C, mean over B/H/W, averaged over L levels)

    All L levels of the teacher are trained against the SAME target E.

    Args:
        teacher: Teacher network to train.
        feature_extractor: Frozen WideResNet50-2 feature extractor.
        dataloader: Training data loader (nominal images only).
        config: Distillation hyperparameters.
        device: Compute device. Default "cuda".
        output_dir: Directory for saving checkpoints. Default "runs".

    Returns:
        Trained teacher network.
    """
    # Set all RNG sources and cuDNN determinism
    set_seed(config.seed, config.deterministic)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Move models to device
    teacher = teacher.to(device)
    feature_extractor = feature_extractor.to(device)

    # Feature extractor is always frozen (eval mode, no gradients).
    # FeatureExtractor.__init__ already calls eval() + requires_grad_(False),
    # but we verify here for safety.
    if feature_extractor.training:
        raise RuntimeError(
            "Feature extractor must be in eval mode for distillation"
        )

    # Teacher enters training mode
    teacher.train()

    # Adam optimizer on teacher parameters only
    optimizer = torch.optim.Adam(
        teacher.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    logger.info(
        "Starting distillation: %d epochs, lr=%.1e, wd=%.1e, batch_size=%d, seed=%d",
        config.num_epochs,
        config.learning_rate,
        config.weight_decay,
        config.batch_size,
        config.seed,
    )

    for epoch in range(config.num_epochs):
        teacher.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Distillation [{epoch + 1}/{config.num_epochs}]",
            leave=False,
        )

        for batch in progress:
            images = batch["image"].to(device)  # (B, 3, 256, 256)

            # Extract distillation target (frozen, no gradients)
            with torch.inference_mode():
                target = feature_extractor(images)  # (B, C, 64, 64)

            # Upsample target to match teacher output resolution (A-009)
            # We clone since target was computed under inference_mode
            target = target.clone()  # (B, C, 64, 64)
            target = _upsample_target(target, (128, 128))  # (B, C, 128, 128)

            # Forward pass through teacher
            teacher_maps = teacher(images)  # L x (B, C, 128, 128)

            # Distillation loss (Eq. 16)
            loss = distillation_loss(target, teacher_maps)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            progress.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info("Epoch %d/%d — avg loss: %.6f", epoch + 1, config.num_epochs, avg_loss)

    # Save trained teacher checkpoint with full metadata for reproducibility.
    # Includes channel_indices from the feature extractor (needed to reproduce
    # the distillation target — see FeatureExtractor docstring).
    #
    # NOTE: model_config is reconstructed from model internals (e.g.
    # teacher.stem.conv.out_channels). This is fragile — if the model
    # architecture changes, these attribute paths break. A cleaner approach
    # would be to pass the full Config (or ModelConfig) into this function
    # and serialize it directly via dataclasses.asdict(). Deferred to a
    # future cleanup (review finding I-6).
    checkpoint_path = output_path / "teacher_distilled.pt"
    checkpoint: dict[str, object] = {
        "model_state_dict": teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "internal_channels": teacher.stem.conv.out_channels,
            "output_channels": teacher.projections[0].out_channels,
            "num_levels": teacher.num_levels,
            "block_depths": tuple(
                len(block) for block in teacher.blocks
            ),
            "kernel_sizes": tuple(
                block[0].conv1.kernel_size[0] for block in teacher.blocks
            ),
        },
        "config": {
            "backbone": config.backbone,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "seed": config.seed,
        },
        "channel_indices": feature_extractor.channel_indices,
        "epoch": config.num_epochs,
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved distilled teacher checkpoint to %s", checkpoint_path)

    return teacher
