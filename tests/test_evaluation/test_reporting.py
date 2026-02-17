"""Tests for mudenet.evaluation.reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mudenet.evaluation.reporting import (
    CategoryResult,
    compute_dataset_averages,
    format_results_table,
    save_results_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_results() -> list[CategoryResult]:
    """Two sample category results without sPRO."""
    return [
        CategoryResult(
            category="bottle", image_auroc=0.99, pixel_auroc=0.98, pro=0.95
        ),
        CategoryResult(
            category="cable", image_auroc=0.95, pixel_auroc=0.92, pro=0.88
        ),
    ]


@pytest.fixture
def sample_results_with_spro() -> list[CategoryResult]:
    """Two sample category results with sPRO (MVTec LOCO)."""
    return [
        CategoryResult(
            category="breakfast_box",
            image_auroc=0.90,
            pixel_auroc=0.85,
            pro=0.80,
            spro=0.75,
        ),
        CategoryResult(
            category="juice_bottle",
            image_auroc=0.88,
            pixel_auroc=0.82,
            pro=0.78,
            spro=0.70,
        ),
    ]


# ---------------------------------------------------------------------------
# CategoryResult
# ---------------------------------------------------------------------------


class TestCategoryResult:
    """Tests for the CategoryResult dataclass."""

    def test_creation(self) -> None:
        """Basic creation with required fields."""
        r = CategoryResult(
            category="bottle", image_auroc=0.99, pixel_auroc=0.98, pro=0.95
        )
        assert r.category == "bottle"
        assert r.image_auroc == 0.99
        assert r.pixel_auroc == 0.98
        assert r.pro == 0.95
        assert r.spro is None

    def test_creation_with_spro(self) -> None:
        """Creation with optional sPRO field."""
        r = CategoryResult(
            category="breakfast_box",
            image_auroc=0.90,
            pixel_auroc=0.85,
            pro=0.80,
            spro=0.75,
        )
        assert r.spro == 0.75

    def test_default_spro_is_none(self) -> None:
        """Default sPRO is None for non-LOCO datasets."""
        r = CategoryResult(category="x", image_auroc=0.5, pixel_auroc=0.5, pro=0.5)
        assert r.spro is None


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------


class TestFormatResultsTable:
    """Tests for table formatting."""

    def test_returns_nonempty_string(
        self, sample_results: list[CategoryResult]
    ) -> None:
        """Table is a non-empty string."""
        table = format_results_table(sample_results, "MVTec AD")
        assert isinstance(table, str)
        assert len(table) > 0

    def test_includes_category_names(
        self, sample_results: list[CategoryResult]
    ) -> None:
        """Table includes all category names."""
        table = format_results_table(sample_results, "MVTec AD")
        assert "bottle" in table
        assert "cable" in table

    def test_includes_mean_row(self, sample_results: list[CategoryResult]) -> None:
        """Table includes a MEAN row."""
        table = format_results_table(sample_results, "MVTec AD")
        assert "MEAN" in table

    def test_includes_dataset_name(
        self, sample_results: list[CategoryResult]
    ) -> None:
        """Table includes the dataset name."""
        table = format_results_table(sample_results, "MVTec AD")
        assert "MVTec AD" in table

    def test_includes_spro_column_when_present(
        self, sample_results_with_spro: list[CategoryResult]
    ) -> None:
        """Table includes sPRO column when results have sPRO values."""
        table = format_results_table(sample_results_with_spro, "MVTec LOCO")
        assert "sPRO" in table

    def test_no_spro_column_when_absent(
        self, sample_results: list[CategoryResult]
    ) -> None:
        """Table omits sPRO column when no results have sPRO."""
        table = format_results_table(sample_results, "MVTec AD")
        assert "sPRO" not in table

    def test_empty_raises(self) -> None:
        """Empty results → ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            format_results_table([], "MVTec AD")


# ---------------------------------------------------------------------------
# save_results_json
# ---------------------------------------------------------------------------


class TestSaveResultsJson:
    """Tests for JSON serialization."""

    def test_creates_file(
        self, tmp_path: Path, sample_results: list[CategoryResult]
    ) -> None:
        """Output file is created."""
        path = tmp_path / "results.json"
        save_results_json(sample_results, path)
        assert path.exists()

    def test_valid_json(
        self, tmp_path: Path, sample_results: list[CategoryResult]
    ) -> None:
        """Output is valid JSON."""
        path = tmp_path / "results.json"
        save_results_json(sample_results, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "results" in data
        assert "averages" in data

    def test_round_trip(
        self, tmp_path: Path, sample_results: list[CategoryResult]
    ) -> None:
        """Write then read back preserves values."""
        path = tmp_path / "results.json"
        save_results_json(sample_results, path)
        data = json.loads(path.read_text(encoding="utf-8"))

        assert len(data["results"]) == 2
        assert data["results"][0]["category"] == "bottle"
        assert data["results"][0]["image_auroc"] == 0.99

    def test_includes_metadata(
        self, tmp_path: Path, sample_results: list[CategoryResult]
    ) -> None:
        """Metadata dict is stored when provided."""
        path = tmp_path / "results.json"
        meta = {"config": "default.yaml", "timestamp": "2026-02-16"}
        save_results_json(sample_results, path, metadata=meta)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["metadata"]["config"] == "default.yaml"

    def test_no_metadata_key_when_none(
        self, tmp_path: Path, sample_results: list[CategoryResult]
    ) -> None:
        """No metadata key when metadata=None."""
        path = tmp_path / "results.json"
        save_results_json(sample_results, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "metadata" not in data

    def test_creates_parent_dirs(
        self, tmp_path: Path, sample_results: list[CategoryResult]
    ) -> None:
        """Creates parent directories if they don't exist."""
        path = tmp_path / "nested" / "dir" / "results.json"
        save_results_json(sample_results, path)
        assert path.exists()

    def test_empty_raises(self, tmp_path: Path) -> None:
        """Empty results → ValueError."""
        path = tmp_path / "results.json"
        with pytest.raises(ValueError, match="must not be empty"):
            save_results_json([], path)


# ---------------------------------------------------------------------------
# compute_dataset_averages
# ---------------------------------------------------------------------------


class TestComputeDatasetAverages:
    """Tests for averaging across categories."""

    def test_known_values(self, sample_results: list[CategoryResult]) -> None:
        """Average of known values is correct."""
        avg = compute_dataset_averages(sample_results)
        assert avg.category == "MEAN"
        assert avg.image_auroc == pytest.approx((0.99 + 0.95) / 2)
        assert avg.pixel_auroc == pytest.approx((0.98 + 0.92) / 2)
        assert avg.pro == pytest.approx((0.95 + 0.88) / 2)

    def test_spro_none_when_all_none(
        self, sample_results: list[CategoryResult]
    ) -> None:
        """Average sPRO is None when no result has sPRO."""
        avg = compute_dataset_averages(sample_results)
        assert avg.spro is None

    def test_spro_averaged_when_present(
        self, sample_results_with_spro: list[CategoryResult]
    ) -> None:
        """Average sPRO is computed when results have sPRO."""
        avg = compute_dataset_averages(sample_results_with_spro)
        assert avg.spro is not None
        assert avg.spro == pytest.approx((0.75 + 0.70) / 2)

    def test_spro_partial_presence(self) -> None:
        """Average sPRO uses only results that have it."""
        results = [
            CategoryResult(category="a", image_auroc=0.9, pixel_auroc=0.8, pro=0.7),
            CategoryResult(
                category="b", image_auroc=0.8, pixel_auroc=0.7, pro=0.6, spro=0.5
            ),
        ]
        avg = compute_dataset_averages(results)
        assert avg.spro == pytest.approx(0.5)

    def test_single_category(self) -> None:
        """Single category → averages equal that category's values."""
        results = [
            CategoryResult(
                category="bottle", image_auroc=0.99, pixel_auroc=0.98, pro=0.95
            ),
        ]
        avg = compute_dataset_averages(results)
        assert avg.image_auroc == 0.99
        assert avg.pixel_auroc == 0.98
        assert avg.pro == 0.95

    def test_empty_raises(self) -> None:
        """Empty results → ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_dataset_averages([])
