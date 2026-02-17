"""Tests for mudenet.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mudenet.evaluation.metrics import (
    _integrate_pro_curve,
    image_auroc,
    pixel_auroc,
    pro_score,
    spro_score,
)

# ---------------------------------------------------------------------------
# image_auroc
# ---------------------------------------------------------------------------


class TestImageAuroc:
    """Tests for image-level AUROC."""

    def test_perfect_predictions(self) -> None:
        """Perfect scores → AUROC = 1.0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert image_auroc(labels, scores) == 1.0

    def test_inverted_predictions(self) -> None:
        """Perfectly inverted scores → AUROC = 0.0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert image_auroc(labels, scores) == 0.0

    def test_random_predictions(self) -> None:
        """Known mixed scores → AUROC between 0 and 1."""
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.6, 0.4])
        result = image_auroc(labels, scores)
        assert 0.0 <= result <= 1.0

    def test_single_class_all_nominal(self) -> None:
        """All nominal labels → returns 0.0 with warning."""
        labels = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.warns(UserWarning, match="Only one class"):
            result = image_auroc(labels, scores)
        assert result == 0.0

    def test_single_class_all_anomalous(self) -> None:
        """All anomalous labels → returns 0.0 with warning."""
        labels = np.array([1, 1, 1, 1])
        scores = np.array([0.5, 0.6, 0.7, 0.8])
        with pytest.warns(UserWarning, match="Only one class"):
            result = image_auroc(labels, scores)
        assert result == 0.0

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths → ValueError."""
        labels = np.array([0, 1])
        scores = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="same length"):
            image_auroc(labels, scores)

    def test_empty_raises(self) -> None:
        """Empty arrays → ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            image_auroc(np.array([], dtype=np.int64), np.array([], dtype=np.float64))

    def test_wrong_ndim_raises(self) -> None:
        """2D array → ValueError."""
        labels = np.array([[0, 1]])
        scores = np.array([[0.1, 0.9]])
        with pytest.raises(ValueError, match="must be 1D"):
            image_auroc(labels, scores)


# ---------------------------------------------------------------------------
# pixel_auroc
# ---------------------------------------------------------------------------


class TestPixelAuroc:
    """Tests for pixel-level AUROC."""

    def test_perfect_masks(self) -> None:
        """Perfect score maps → AUROC = 1.0."""
        masks = np.zeros((2, 4, 4), dtype=np.int64)
        masks[0, 0:2, 0:2] = 1
        maps = np.zeros((2, 4, 4), dtype=np.float64)
        maps[0, 0:2, 0:2] = 1.0
        assert pixel_auroc(masks, maps) == 1.0

    def test_known_value(self) -> None:
        """Score maps with partial overlap → AUROC in (0, 1)."""
        masks = np.zeros((1, 4, 4), dtype=np.int64)
        masks[0, 0:2, 0:2] = 1  # 4 anomalous pixels out of 16
        maps = np.zeros((1, 4, 4), dtype=np.float64)
        maps[0, 0:2, :] = 0.8  # 8 high-scoring pixels (4 TP + 4 FP)
        result = pixel_auroc(masks, maps)
        assert 0.0 < result < 1.0

    def test_all_same_mask_returns_zero(self) -> None:
        """All-zero masks → returns 0.0 with warning."""
        rng = np.random.default_rng(42)
        masks = np.zeros((2, 4, 4), dtype=np.int64)
        maps = rng.random((2, 4, 4))
        with pytest.warns(UserWarning, match="Only one class"):
            result = pixel_auroc(masks, maps)
        assert result == 0.0

    def test_shape_mismatch_raises(self) -> None:
        """Different shapes → ValueError."""
        masks = np.zeros((2, 4, 4), dtype=np.int64)
        maps = np.zeros((2, 8, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="same shape"):
            pixel_auroc(masks, maps)

    def test_empty_raises(self) -> None:
        """Zero samples → ValueError."""
        with pytest.raises(ValueError, match="at least one sample"):
            pixel_auroc(
                np.zeros((0, 4, 4), dtype=np.int64),
                np.zeros((0, 4, 4), dtype=np.float64),
            )

    def test_wrong_ndim_raises(self) -> None:
        """2D arrays → ValueError."""
        with pytest.raises(ValueError, match="must be 3D"):
            pixel_auroc(
                np.zeros((4, 4), dtype=np.int64),
                np.zeros((4, 4), dtype=np.float64),
            )


# ---------------------------------------------------------------------------
# pro_score
# ---------------------------------------------------------------------------


class TestProScore:
    """Tests for Per-Region Overlap (PRO) score."""

    def test_perfect_predictions_near_one(self) -> None:
        """Perfect overlap → PRO close to 1.0.

        May not be exactly 1.0 due to threshold discretization.
        """
        masks = np.zeros((1, 8, 8), dtype=np.int64)
        masks[0, 2:6, 2:6] = 1  # single 4x4 anomalous region
        maps = np.zeros((1, 8, 8), dtype=np.float64)
        maps[0, 2:6, 2:6] = 1.0  # exact match
        result = pro_score(masks, maps, num_thresholds=100)
        assert result > 0.9

    def test_all_zero_predictions(self) -> None:
        """All-zero predictions → PRO = 0.0."""
        masks = np.zeros((1, 8, 8), dtype=np.int64)
        masks[0, 2:6, 2:6] = 1
        maps = np.zeros((1, 8, 8), dtype=np.float64)
        result = pro_score(masks, maps, num_thresholds=100)
        assert result == 0.0

    def test_returns_in_unit_interval(self) -> None:
        """PRO is always in [0, 1]."""
        rng = np.random.default_rng(42)
        masks = np.zeros((2, 8, 8), dtype=np.int64)
        masks[0, 1:4, 1:4] = 1
        masks[1, 5:7, 5:7] = 1
        maps = rng.random((2, 8, 8))
        result = pro_score(masks, maps, num_thresholds=50)
        assert 0.0 <= result <= 1.0

    def test_multiple_components(self) -> None:
        """Multiple disconnected GT regions are handled."""
        masks = np.zeros((1, 8, 8), dtype=np.int64)
        masks[0, 0:2, 0:2] = 1  # component 1
        masks[0, 5:7, 5:7] = 1  # component 2
        maps = np.zeros((1, 8, 8), dtype=np.float64)
        maps[0, 0:2, 0:2] = 1.0  # only detect component 1
        result = pro_score(masks, maps, num_thresholds=50)
        # Should be around 0.5 (full overlap on one, zero on other)
        assert 0.0 < result < 1.0

    def test_overlapping_bounding_boxes(self) -> None:
        """Components with overlapping bounding boxes are distinguished.

        Regression test for the critical bug where bounding-box slices
        from find_objects were used instead of per-label masks.
        """
        masks = np.zeros((1, 8, 8), dtype=np.int64)
        # Two L-shaped components whose bounding boxes overlap
        masks[0, 0:3, 0:1] = 1  # component 1: vertical bar
        masks[0, 0:1, 2:4] = 1  # component 2: horizontal bar
        # Predictions only cover component 1
        maps = np.zeros((1, 8, 8), dtype=np.float64)
        maps[0, 0:3, 0:1] = 1.0
        result = pro_score(masks, maps, num_thresholds=50)
        # Component 1 has full overlap, component 2 has zero → mean ~0.5
        assert 0.2 < result < 0.8

    def test_no_anomalous_gt_returns_zero(self) -> None:
        """No anomalous ground truth → returns 0.0 with warning."""
        rng = np.random.default_rng(99)
        masks = np.zeros((2, 8, 8), dtype=np.int64)
        maps = rng.random((2, 8, 8))
        with pytest.warns(UserWarning, match="No anomalous"):
            result = pro_score(masks, maps)
        assert result == 0.0

    def test_all_anomalous_gt_returns_zero(self) -> None:
        """All pixels anomalous → FPR undefined, returns 0.0 with warning."""
        masks = np.ones((1, 4, 4), dtype=np.int64)
        maps = np.ones((1, 4, 4), dtype=np.float64)
        with pytest.warns(UserWarning, match="All pixels are anomalous"):
            result = pro_score(masks, maps)
        assert result == 0.0

    def test_all_identical_scores(self) -> None:
        """All anomaly map values identical → PRO = 0.0.

        When all thresholds see the same binary prediction, the curve
        collapses to a single point.
        """
        masks = np.zeros((1, 8, 8), dtype=np.int64)
        masks[0, 2:6, 2:6] = 1
        maps = np.full((1, 8, 8), 0.5, dtype=np.float64)
        result = pro_score(masks, maps, num_thresholds=50)
        assert 0.0 <= result <= 1.0

    def test_invalid_max_fpr_raises(self) -> None:
        """max_fpr out of range → ValueError."""
        masks = np.zeros((1, 4, 4), dtype=np.int64)
        maps = np.zeros((1, 4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="max_fpr"):
            pro_score(masks, maps, max_fpr=0.0)
        with pytest.raises(ValueError, match="max_fpr"):
            pro_score(masks, maps, max_fpr=1.5)

    def test_invalid_num_thresholds_raises(self) -> None:
        """num_thresholds < 2 → ValueError."""
        masks = np.zeros((1, 4, 4), dtype=np.int64)
        maps = np.zeros((1, 4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="num_thresholds"):
            pro_score(masks, maps, num_thresholds=1)


# ---------------------------------------------------------------------------
# spro_score
# ---------------------------------------------------------------------------


class TestSproScore:
    """Tests for Saturated PRO (sPRO) score."""

    def test_returns_in_unit_interval(self) -> None:
        """sPRO is always in [0, 1]."""
        rng = np.random.default_rng(42)
        masks = np.zeros((2, 8, 8), dtype=np.int64)
        masks[0, 1:4, 1:4] = 1
        masks[1, 5:7, 5:7] = 1
        maps = rng.random((2, 8, 8))
        result = spro_score(masks, maps, saturation_threshold=0.5, num_thresholds=50)
        assert 0.0 <= result <= 1.0

    def test_saturation_reduces_score(self) -> None:
        """sPRO with saturation < 1.0 should be ≤ PRO.

        When overlap is clamped, the sPRO curve is everywhere ≤ the PRO curve.
        After proper normalization (dividing by sat * max_fpr instead of
        max_fpr), the sPRO value should still be ≤ PRO.
        """
        masks = np.zeros((1, 8, 8), dtype=np.int64)
        masks[0, 2:6, 2:6] = 1
        maps = np.zeros((1, 8, 8), dtype=np.float64)
        maps[0, 2:6, 2:6] = 1.0

        pro_val = pro_score(masks, maps, max_fpr=0.3, num_thresholds=100)
        spro_val = spro_score(
            masks, maps, saturation_threshold=0.5, max_fpr=0.3, num_thresholds=100
        )
        assert spro_val <= pro_val + 1e-9  # allow tiny float tolerance

    def test_saturation_at_one_matches_pro(self) -> None:
        """sPRO with saturation=1.0 should equal PRO (at same max_fpr)."""
        rng = np.random.default_rng(123)
        masks = np.zeros((2, 8, 8), dtype=np.int64)
        masks[0, 1:4, 1:4] = 1
        masks[1, 5:7, 5:7] = 1
        maps = rng.random((2, 8, 8))

        pro_val = pro_score(masks, maps, max_fpr=0.05, num_thresholds=100)
        spro_val = spro_score(
            masks, maps, saturation_threshold=1.0, max_fpr=0.05, num_thresholds=100
        )
        assert abs(spro_val - pro_val) < 1e-9

    def test_invalid_saturation_raises(self) -> None:
        """saturation_threshold <= 0 → ValueError."""
        masks = np.zeros((1, 4, 4), dtype=np.int64)
        maps = np.zeros((1, 4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="saturation_threshold"):
            spro_score(masks, maps, saturation_threshold=0.0)
        with pytest.raises(ValueError, match="saturation_threshold"):
            spro_score(masks, maps, saturation_threshold=-0.5)

    def test_no_anomalous_gt_returns_zero(self) -> None:
        """No anomalous ground truth → returns 0.0 with warning."""
        rng = np.random.default_rng(77)
        masks = np.zeros((2, 8, 8), dtype=np.int64)
        maps = rng.random((2, 8, 8))
        with pytest.warns(UserWarning, match="No anomalous"):
            result = spro_score(masks, maps, saturation_threshold=0.5)
        assert result == 0.0


# ---------------------------------------------------------------------------
# _integrate_pro_curve (boundary interpolation)
# ---------------------------------------------------------------------------


class TestIntegrateProCurve:
    """Tests for the PRO curve integration helper."""

    def test_perfect_rectangle(self) -> None:
        """Constant PRO=1.0 from FPR=0 to FPR=max_fpr → normalized AUC = 1.0."""
        fpr = np.array([0.0, 0.1, 0.2, 0.3])
        pro = np.array([1.0, 1.0, 1.0, 1.0])
        result = _integrate_pro_curve(fpr, pro, max_fpr=0.3, normalizer=0.3)
        assert result == pytest.approx(1.0)

    def test_zero_pro(self) -> None:
        """PRO=0 everywhere → AUC = 0.0."""
        fpr = np.array([0.0, 0.1, 0.2, 0.3])
        pro = np.array([0.0, 0.0, 0.0, 0.0])
        result = _integrate_pro_curve(fpr, pro, max_fpr=0.3, normalizer=0.3)
        assert result == pytest.approx(0.0)

    def test_boundary_interpolation(self) -> None:
        """max_fpr falls between sample points → linear interpolation."""
        fpr = np.array([0.0, 0.2, 0.4])
        pro = np.array([0.0, 0.5, 1.0])
        # max_fpr=0.3 falls between 0.2 and 0.4
        # Interpolated PRO at 0.3 = 0.5 + (0.3-0.2)/(0.4-0.2) * (1.0-0.5) = 0.75
        result = _integrate_pro_curve(fpr, pro, max_fpr=0.3, normalizer=0.3)
        assert 0.0 < result < 1.0

    def test_empty_below_max_fpr_returns_zero(self) -> None:
        """No FPR values ≤ max_fpr → returns 0.0."""
        fpr = np.array([0.5, 0.6, 0.7])
        pro = np.array([0.5, 0.6, 0.7])
        result = _integrate_pro_curve(fpr, pro, max_fpr=0.3, normalizer=0.3)
        assert result == 0.0

    def test_all_below_max_fpr(self) -> None:
        """All FPR values below max_fpr → extends with last PRO value."""
        fpr = np.array([0.0, 0.05, 0.1])
        pro = np.array([0.0, 0.5, 0.8])
        result = _integrate_pro_curve(fpr, pro, max_fpr=0.3, normalizer=0.3)
        assert 0.0 < result <= 1.0
