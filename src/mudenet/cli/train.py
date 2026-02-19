"""CLI subcommand for Stage 2: end-to-end training.

Trains S1, A, S2 jointly with frozen teacher T (Eq. 8, Sec. 4).

Usage:
    mudenet train --config configs/mvtec_ad.yaml --category bottle \
        --checkpoint runs/teacher_distilled.pt [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from mudenet.cli.utils import (
    add_common_args,
    create_autoencoder_from_config,
    create_dataset_and_loader,
    create_teacher_from_config,
    load_config_from_subcommand,
)

logger = logging.getLogger(__name__)


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'train' subcommand.

    Args:
        subparsers: Subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        "train",
        help="Stage 2: end-to-end training",
        description=(
            "Train S1, autoencoder, and S2 jointly with a frozen distilled "
            "teacher T (Eq. 8, Sec. 4)."
        ),
    )
    add_common_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to distilled teacher checkpoint (teacher_distilled.pt)",
    )
    parser.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> None:
    """Execute Stage 2: end-to-end training.

    Workflow:
        1. Load config with CLI overrides
        2. Load distilled teacher checkpoint and create all 4 models
        3. Freeze teacher (eval + no gradients)
        4. Create training dataset and dataloader
        5. Run end-to-end training loop
        6. Save trained checkpoint

    Args:
        args: Parsed CLI arguments.

    Raises:
        SystemExit: On file not found, missing keys, or runtime errors.
    """
    try:
        config = load_config_from_subcommand(args, seed_target="training.seed")

        logger.info(
            "Starting end-to-end training — dataset: %s, category: %s, device: %s",
            config.data.dataset_type,
            config.data.category,
            config.device,
        )

        from mudenet.training.trainer import train_end_to_end

        # Load distilled teacher checkpoint
        logger.info("Loading distilled teacher from %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

        # Create teacher and load distilled weights
        teacher = create_teacher_from_config(config)
        teacher.load_state_dict(checkpoint["model_state_dict"])

        # Freeze teacher: eval mode + no gradients
        teacher.eval()
        teacher.requires_grad_(False)

        # Create fresh student1 (same architecture as teacher)
        student1 = create_teacher_from_config(config)

        # Create fresh autoencoder
        autoencoder = create_autoencoder_from_config(config)

        # Create fresh student2 (same architecture as teacher)
        student2 = create_teacher_from_config(config)

        # Create training dataset and dataloader
        _, dataloader = create_dataset_and_loader(
            config,
            split="train",
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
        )

        # Run end-to-end training (sets seed internally)
        train_end_to_end(
            teacher,
            student1,
            autoencoder,
            student2,
            dataloader,
            config.training,
            device=config.device,
            output_dir=config.output_dir,
        )

        logger.info(
            "Training complete. Checkpoint saved to %s/end_to_end.pt",
            config.output_dir,
        )

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except KeyError as e:
        logger.error("Missing key in checkpoint or config: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        sys.exit(1)
