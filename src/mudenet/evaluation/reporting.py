"""Result aggregation and formatted reporting.

Provides structured storage and display of per-category evaluation results,
matching the paper's table format (Tables 2-9).

Classes:
    - CategoryResult: Per-category evaluation result dataclass
Functions:
    - format_results_table: Human-readable aligned table
    - save_results_json: JSON serialization with optional metadata
    - compute_dataset_averages: Mean metrics across categories
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CategoryResult:
    """Evaluation results for a single category.

    Attributes:
        category: Category name (e.g. "bottle").
        image_auroc: Image-level AUROC.
        pixel_auroc: Pixel-level AUROC, or None if masks unavailable.
        pro: PRO score (normalized AUPRO at 30% FPR), or None if not
            applicable.
        spro: sPRO score (normalized AUsPRO at 5% FPR), or None if not
            applicable (MVTec AD, VisA).
    """

    category: str
    image_auroc: float
    pixel_auroc: float | None = None
    pro: float | None = None
    spro: float | None = None


def format_results_table(
    results: list[CategoryResult],
    dataset_name: str,
) -> str:
    """Format results as a human-readable table matching paper format (Tables 2-9).

    Includes a "MEAN" row at the bottom computed from the provided results.

    Args:
        results: Per-category results.
        dataset_name: Dataset name for the table header.

    Returns:
        Formatted table string with aligned columns.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("results must not be empty")

    has_spro = any(r.spro is not None for r in results)
    averages = compute_dataset_averages(results)

    # Build header
    headers = ["Category", "I-AUROC", "P-AUROC", "PRO"]
    if has_spro:
        headers.append("sPRO")

    # Build rows
    rows: list[list[str]] = []
    for r in results:
        row = [r.category, _fmt(r.image_auroc), _fmt(r.pixel_auroc), _fmt(r.pro)]
        if has_spro:
            row.append(_fmt(r.spro))
        rows.append(row)

    # Add MEAN row
    mean_row = [
        averages.category,
        _fmt(averages.image_auroc),
        _fmt(averages.pixel_auroc),
        _fmt(averages.pro),
    ]
    if has_spro:
        mean_row.append(_fmt(averages.spro))
    rows.append(mean_row)

    # Compute column widths
    all_rows = [headers, *rows]
    col_widths = [
        max(len(cell) for cell in col) for col in zip(*all_rows, strict=True)
    ]

    # Format table
    lines: list[str] = []
    title = f"  {dataset_name} Results"
    lines.append(title)
    lines.append("=" * max(len(title), sum(col_widths) + 3 * (len(headers) - 1)))

    # Header row
    header_line = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths, strict=True)
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    for i, row in enumerate(rows):
        line = " | ".join(
            cell.ljust(w) for cell, w in zip(row, col_widths, strict=True)
        )
        # Separator before MEAN row
        if i == len(rows) - 1:
            lines.append("-" * len(header_line))
        lines.append(line)

    return "\n".join(lines)


def save_results_json(
    results: list[CategoryResult],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save per-category results as JSON for programmatic consumption.

    Args:
        results: Per-category results.
        output_path: Output file path.
        metadata: Optional metadata dict (config, timestamp, etc.).

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("results must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    averages = compute_dataset_averages(results)

    data: dict[str, Any] = {
        "results": [asdict(r) for r in results],
        "averages": asdict(averages),
    }
    if metadata is not None:
        data["metadata"] = metadata

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", output_path)


def compute_dataset_averages(
    results: list[CategoryResult],
) -> CategoryResult:
    """Compute mean metrics across categories.

    Args:
        results: Per-category results.

    Returns:
        CategoryResult with category="MEAN" and averaged metrics.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("results must not be empty")

    n = len(results)
    mean_image = sum(r.image_auroc for r in results) / n

    # Average pixel-level metrics only if at least one result has them
    pixel_values = [r.pixel_auroc for r in results if r.pixel_auroc is not None]
    mean_pixel: float | None = None
    if pixel_values:
        mean_pixel = sum(pixel_values) / len(pixel_values)

    pro_values = [r.pro for r in results if r.pro is not None]
    mean_pro: float | None = None
    if pro_values:
        mean_pro = sum(pro_values) / len(pro_values)

    spro_values = [r.spro for r in results if r.spro is not None]
    mean_spro: float | None = None
    if spro_values:
        mean_spro = sum(spro_values) / len(spro_values)

    return CategoryResult(
        category="MEAN",
        image_auroc=mean_image,
        pixel_auroc=mean_pixel,
        pro=mean_pro,
        spro=mean_spro,
    )


def _fmt(value: float | None) -> str:
    """Format a metric value as a percentage string with 1 decimal place."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}"
