"""CLI entry point for MuDeNet.

Dispatches to subcommands::

    mudenet distill  — Stage 1: pre-knowledge distillation
    mudenet train    — Stage 2: end-to-end training
    mudenet evaluate — Evaluate trained model on test set

Usage:
    python -m mudenet distill --config configs/default.yaml
    python -m mudenet train --config configs/mvtec_ad.yaml --category bottle --checkpoint runs/teacher_distilled.pt
    python -m mudenet evaluate --config configs/mvtec_ad.yaml --category bottle --checkpoint runs/end_to_end.pt
"""

from __future__ import annotations

import argparse
import logging
import sys

from mudenet.cli.distill import add_distill_parser
from mudenet.cli.evaluate import add_evaluate_parser
from mudenet.cli.train import add_train_parser

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point.

    Dispatches to subcommands:
        mudenet distill  — Stage 1: pre-knowledge distillation
        mudenet train    — Stage 2: end-to-end training
        mudenet evaluate — Evaluate trained model on test set

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

    parser = argparse.ArgumentParser(
        prog="mudenet",
        description=(
            "MuDeNet — Multi-patch descriptor network for "
            "visual anomaly detection and segmentation"
        ),
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        description="Available commands",
    )

    add_distill_parser(subparsers)
    add_train_parser(subparsers)
    add_evaluate_parser(subparsers)

    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
