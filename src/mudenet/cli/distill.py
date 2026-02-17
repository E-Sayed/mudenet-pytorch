"""CLI subcommand for Stage 1: pre-knowledge distillation.

Trains the teacher network T by distilling from WideResNet50-2 (Eq. 16).

Usage:
    mudenet distill --config configs/default.yaml [--category bottle] [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import sys

from mudenet.cli.utils import (
    add_common_args,
    create_dataset_and_loader,
    create_teacher_from_config,
    load_config_from_subcommand,
)

logger = logging.getLogger(__name__)


def add_distill_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'distill' subcommand.

    Args:
        subparsers: Subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        "distill",
        help="Stage 1: pre-knowledge distillation",
        description=(
            "Train the teacher network T by distilling from a pretrained "
            "WideResNet50-2 feature extractor (Eq. 16, Sec. 3.3)."
        ),
    )
    add_common_args(parser)
    parser.set_defaults(func=run_distill)


def run_distill(args: argparse.Namespace) -> None:
    """Execute Stage 1: pre-knowledge distillation.

    Workflow:
        1. Load config with CLI overrides
        2. Create FeatureExtractor (frozen, pretrained backbone)
        3. Create TeacherNetwork (to be trained)
        4. Create training dataset and dataloader
        5. Run distillation training loop
        6. Save trained teacher checkpoint

    Args:
        args: Parsed CLI arguments.
    """
    try:
        config = load_config_from_subcommand(args, seed_target="distillation.seed")

        logger.info(
            "Starting distillation — dataset: %s, category: %s, device: %s",
            config.data.dataset_type,
            config.data.category,
            config.device,
        )

        from mudenet.models.feature_extractor import FeatureExtractor
        from mudenet.training.distillation import train_distillation

        # Create feature extractor (frozen, pretrained WideResNet50-2)
        fe = FeatureExtractor(
            output_channels=config.model.num_channels,
            seed=config.distillation.seed,
        )

        # Create teacher network (to be trained)
        teacher = create_teacher_from_config(config)

        # Create training dataset and dataloader.
        # DistillationConfig has no num_workers — use training.num_workers
        # (same machine, same I/O constraints).
        _, dataloader = create_dataset_and_loader(
            config,
            split="train",
            batch_size=config.distillation.batch_size,
            num_workers=config.training.num_workers,
            seed=config.distillation.seed,
        )

        # Run distillation (sets seed internally)
        train_distillation(
            teacher,
            fe,
            dataloader,
            config.distillation,
            device=config.device,
            output_dir=config.output_dir,
        )

        logger.info(
            "Distillation complete. Teacher checkpoint saved to %s/teacher_distilled.pt",
            config.output_dir,
        )

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except KeyError as e:
        logger.error("Missing configuration key: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        sys.exit(1)
