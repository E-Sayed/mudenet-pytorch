"""CLI subcommand for evaluation.

Evaluates a trained MuDeNet model on a test set, computing image-level and
pixel-level anomaly detection metrics.

Usage:
    mudenet evaluate --config configs/mvtec_ad.yaml --category bottle \
        --checkpoint runs/end_to_end.pt [--device cuda] [--output-dir results]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from mudenet.cli.utils import (
    add_common_args,
    create_autoencoder_from_config,
    create_dataset_and_loader,
    create_teacher_from_config,
    get_dataset_class,
    load_config_from_subcommand,
)
from mudenet.config.schema import Config
from mudenet.data.utils import create_dataloader

logger = logging.getLogger(__name__)


def add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'evaluate' subcommand.

    Args:
        subparsers: Subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model on test set",
        description=(
            "Evaluate a trained MuDeNet model on the test split. Computes "
            "image-AUROC, pixel-AUROC, PRO, and sPRO (MVTec LOCO only)."
        ),
    )
    add_common_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to end-to-end training checkpoint (end_to_end.pt)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Save per-image visualizations (original | GT mask | anomaly heatmap)",
    )
    parser.add_argument(
        "--num-visualize",
        type=int,
        default=0,
        help=(
            "Max number of visualizations to save (0 = all test images). "
            "Only used when --visualize is set."
        ),
    )
    parser.set_defaults(func=run_evaluate)


def _create_validation_loader(
    config: Config,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a validation dataloader for normalization statistics.

    Uses a random subset of the training data (controlled by
    ``config.inference.validation_ratio``) with eval transforms
    (deterministic resize + normalize, no augmentations) and a seeded
    split for reproducibility.

    Args:
        config: Full configuration.

    Returns:
        DataLoader for the validation subset.
    """
    from mudenet.data.transforms import get_eval_transform

    # Load training images with eval transforms (no augmentations) so
    # normalization statistics are computed on deterministic transforms
    # matching test time.  We use get_eval_transform instead of
    # get_train_transform to avoid random flips/rotation/color jitter.
    dataset_cls = get_dataset_class(config.data.dataset_type)
    eval_transform = get_eval_transform(config.data)

    train_dataset = dataset_cls(
        data_root=config.data.data_root,
        category=config.data.category,
        split="train",
        transform=eval_transform,
        target_transform=None,
    )

    # Split into (unused_train, validation) with seeded generator
    total = len(train_dataset)
    val_size = max(1, int(total * config.inference.validation_ratio))
    train_size = total - val_size

    generator = torch.Generator().manual_seed(config.training.seed)
    _, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    logger.info(
        "Validation split: %d samples (%.0f%% of %d training samples)",
        val_size,
        config.inference.validation_ratio * 100,
        total,
    )

    return create_dataloader(
        val_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
        pin_memory=(config.device != "cpu"),
    )


def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained model on test set.

    Workflow:
        1. Load config with CLI overrides
        2. Load checkpoint and create all 4 models
        3. Freeze all models
        4. Create test dataset and dataloader
        5. Compute normalization stats on validation split
        6. Run inference on test set, collecting predictions
        7. Compute metrics (image-AUROC, pixel-AUROC, PRO, sPRO)
        8. Display and save results

    Args:
        args: Parsed CLI arguments.

    Raises:
        SystemExit: On file not found, missing keys, or runtime errors.
    """
    try:
        if args.category is None:
            logger.error(
                "--category is required for evaluation. "
                "Specify the dataset category (e.g. --category bottle)."
            )
            sys.exit(1)

        config = load_config_from_subcommand(args, seed_target="training.seed")

        logger.info(
            "Starting evaluation — dataset: %s, category: %s, device: %s",
            config.data.dataset_type,
            config.data.category,
            config.device,
        )

        import torch.nn.functional as F

        from mudenet.evaluation.metrics import (
            image_auroc,
            pixel_auroc,
            pro_score,
            spro_score,
        )
        from mudenet.evaluation.reporting import (
            CategoryResult,
            format_results_table,
            save_results_json,
        )
        from mudenet.inference.pipeline import (
            compute_image_score,
            compute_normalization_stats,
            score_batch,
        )
        from mudenet.utils.seed import set_seed

        # Configure cuDNN determinism and seed RNGs for evaluation
        set_seed(config.training.seed, config.training.deterministic)

        # Load end-to-end training checkpoint
        logger.info("Loading checkpoint from %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

        # Create all 4 models from config
        teacher = create_teacher_from_config(config)
        student1 = create_teacher_from_config(config)
        autoencoder = create_autoencoder_from_config(config)
        student2 = create_teacher_from_config(config)

        # Load weights from checkpoint
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
        student1.load_state_dict(checkpoint["student1_state_dict"])
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        student2.load_state_dict(checkpoint["student2_state_dict"])

        # Freeze all models: eval mode + no gradients + move to device
        for model in (teacher, student1, autoencoder, student2):
            model.eval()
            model.requires_grad_(False)
            model.to(config.device)

        # Create test dataset and dataloader (with mask transforms)
        _, test_loader = create_dataset_and_loader(
            config,
            split="test",
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
        )

        # Compute normalization stats on validation split of training data
        logger.info("Computing normalization statistics on validation split...")
        val_loader = _create_validation_loader(config)
        norm_stats = compute_normalization_stats(
            teacher,
            student1,
            autoencoder,
            student2,
            val_loader,
            device=config.device,
        )
        logger.info("Normalization stats computed successfully")

        # Run inference on test set
        logger.info("Running inference on test set...")
        all_scores: list[float] = []
        all_labels: list[int] = []
        all_anomaly_maps: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        all_images: list[torch.Tensor] = []  # keep for visualization
        all_paths: list[str] = []

        with torch.inference_mode():
            for batch in test_loader:
                images = batch["image"].to(config.device)  # (B, 3, 256, 256)
                labels = batch["label"]  # (B,)
                masks = batch["mask"]  # (B, 1, H, W) or None entries

                # Compute anomaly maps and image scores
                anomaly_map = score_batch(
                    teacher, student1, autoencoder, student2,
                    images, norm_stats,
                )  # (B, 128, 128)

                # Upsample anomaly map to input resolution so it matches
                # ground-truth masks which are resized to image_size.
                anomaly_map = F.interpolate(
                    anomaly_map.unsqueeze(1),  # (B, 1, 128, 128)
                    size=(config.data.image_size, config.data.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # (B, image_size, image_size)

                image_scores = compute_image_score(anomaly_map)  # (B,)

                # Collect predictions
                all_scores.extend(image_scores.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_anomaly_maps.append(anomaly_map.cpu().numpy())

                # All samples have mask tensors (zero for nominal, non-zero
                # for anomalous) — no None guard needed after C-2 fix.
                all_masks.append(masks.numpy())

                # Store originals for visualization
                if args.visualize:
                    all_images.append(images.cpu())
                    all_paths.extend(batch["path"])

        # Convert to numpy arrays
        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)
        anomaly_maps_arr = np.concatenate(all_anomaly_maps, axis=0)  # (N, H, W)

        # Compute image-level AUROC
        img_auc = image_auroc(labels_arr, scores_arr)
        logger.info("Image AUROC: %.4f", img_auc)

        # Compute pixel-level metrics (require ground-truth masks)
        pix_auc: float | None = None
        pro: float | None = None
        spro: float | None = None

        if all_masks:
            masks_arr = np.concatenate(all_masks, axis=0)  # (N, 1, H, W)
            # Squeeze channel dim: (N, 1, H, W) -> (N, H, W)
            if masks_arr.ndim == 4 and masks_arr.shape[1] == 1:
                masks_arr = masks_arr[:, 0]

            pix_auc = pixel_auroc(masks_arr, anomaly_maps_arr)
            logger.info("Pixel AUROC: %.4f", pix_auc)

            # PRO score — for MVTec AD and VisA
            if config.data.dataset_type in ("mvtec_ad", "visa"):
                pro = pro_score(masks_arr, anomaly_maps_arr)
                logger.info("PRO: %.4f", pro)

            # sPRO score — for MVTec LOCO only
            if config.data.dataset_type == "mvtec_loco":
                spro = spro_score(masks_arr, anomaly_maps_arr)
                logger.info("sPRO: %.4f", spro)
        else:
            logger.warning(
                "No ground-truth masks found — skipping pixel-level metrics"
            )

        # Create result and display
        result = CategoryResult(
            category=config.data.category,
            image_auroc=img_auc,
            pixel_auroc=pix_auc,
            pro=pro,
            spro=spro,
        )

        table = format_results_table(
            [result],
            dataset_name=config.data.dataset_type,
        )
        logger.info("Results:\n%s", table)

        # Save results to JSON
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / f"{config.data.category}_results.json"
        save_results_json(
            [result],
            results_file,
            metadata={
                "config": args.config,
                "checkpoint": args.checkpoint,
                "category": config.data.category,
                "dataset_type": config.data.dataset_type,
            },
        )
        logger.info("Results saved to %s", results_file)

        # Generate visualizations if requested
        if args.visualize:
            from mudenet.visualization.overlay import (
                denormalize_image,
                save_visualization,
            )

            vis_dir = output_path / f"{config.data.category}_visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

            images_tensor = torch.cat(all_images, dim=0)  # (N, 3, H, W)
            num_vis = len(images_tensor)
            if args.num_visualize > 0:
                num_vis = min(args.num_visualize, num_vis)

            logger.info(
                "Saving %d visualizations to %s ...", num_vis, vis_dir,
            )

            for i in range(num_vis):
                img_np = denormalize_image(images_tensor[i])  # (H, W, 3) uint8
                amap = anomaly_maps_arr[i]  # (H, W)

                # Get ground truth mask (if any anomalous pixels)
                gt_mask = None
                if all_masks:
                    m = masks_arr[i]  # (H, W)
                    if m.max() > 0:
                        gt_mask = (m > 0).astype(np.uint8)

                # Derive filename from original path
                src = Path(all_paths[i])
                defect_type = src.parent.name
                stem = src.stem
                label_str = "GOOD" if all_labels[i] == 0 else "ANOMALY"
                score_str = f"{all_scores[i]:.3f}"
                vis_name = f"{defect_type}_{stem}_{label_str}_s{score_str}.png"

                save_visualization(
                    image=img_np,
                    anomaly_map=amap,
                    output_path=vis_dir / vis_name,
                    ground_truth_mask=gt_mask,
                    colormap="jet",
                    alpha=0.5,
                )

            logger.info("Visualizations saved to %s", vis_dir)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except KeyError as e:
        logger.error("Missing key in checkpoint or config: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        sys.exit(1)
