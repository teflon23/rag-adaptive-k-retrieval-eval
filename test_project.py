"""Tests for adaptive-k vs fixed-k RAG retrieval context selection."""

from __future__ import annotations

import numpy as np
import pytest

from app import run_experiment
from data import make_dataset


class TestDataset:
    """Tests for the synthetic dataset generator."""

    def test_default_shape_and_types(self) -> None:
        """make_dataset returns X and y with correct length and types."""
        X, y = make_dataset(seed=42, n_samples=256)
        assert len(X) == 256
        assert len(y) == 256
        assert X.dtype == object
        assert y.dtype == np.int64

    def test_different_seed_changes_queries(self) -> None:
        """Different seeds must produce different query arrays."""
        X1, _ = make_dataset(seed=42, n_samples=256)
        X2, _ = make_dataset(seed=7, n_samples=256)
        assert not np.array_equal(X1, X2)

    def test_invalid_n_samples_raises(self) -> None:
        """n_samples < 32 must raise ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 32"):
            make_dataset(seed=42, n_samples=16)

    def test_n_samples_respected(self) -> None:
        """Different n_samples values must be respected."""
        X, y = make_dataset(seed=42, n_samples=64)
        assert len(X) == 64
        assert len(y) == 64


class TestExperiment:
    """Tests for the run_experiment function."""

    def test_baseline_avg_context_size_is_three(self) -> None:
        """Baseline fixed top-3 must always have avg_context_size == 3.0."""
        result = run_experiment(seed=42, n_samples=256)
        assert result["baseline_metrics"]["avg_context_size"] == 3.0

    def test_adaptive_avg_context_size_at_least_one(self) -> None:
        """Adaptive-k must select at least 1 doc per query (fallback to top-1)."""
        result = run_experiment(seed=42, n_samples=256)
        assert result["metrics"]["avg_context_size"] >= 1.0

    def test_determinism(self) -> None:
        """Two identical calls must return exactly equal dicts."""
        r1 = run_experiment(seed=42, n_samples=256)
        r2 = run_experiment(seed=42, n_samples=256)
        assert r1 == r2

    def test_n_samples_respected_in_output(self) -> None:
        """Different n_samples must be reflected in the output."""
        result = run_experiment(seed=42, n_samples=128)
        assert result["n_samples"] == 128

    def test_metrics_are_finite(self) -> None:
        """All metric values must be finite numbers."""
        result = run_experiment(seed=42, n_samples=256)
        for key in ("metrics", "baseline_metrics"):
            for name, value in result[key].items():
                assert isinstance(value, float), f"{key}.{name} is not float"
                assert np.isfinite(value), f"{key}.{name} is not finite"

    def test_hit_at_k_bounded_by_one(self) -> None:
        """hit_at_k must be in [0, 1] for both methods."""
        result = run_experiment(seed=42, n_samples=256)
        for key in ("metrics", "baseline_metrics"):
            assert 0.0 <= result[key]["hit_at_k"] <= 1.0
            assert 0.0 <= result[key]["hit_at_1"] <= 1.0

    def test_hit_at_1_leq_hit_at_k(self) -> None:
        """hit_at_1 (gold ranked #1) implies gold is in selected set, so hit_at_1 <= hit_at_k."""
        result = run_experiment(seed=42, n_samples=256)
        for key in ("metrics", "baseline_metrics"):
            assert result[key]["hit_at_1"] <= result[key]["hit_at_k"] + 1e-12

    def test_output_structure(self) -> None:
        """run_experiment must return the required top-level keys."""
        result = run_experiment(seed=42, n_samples=256)
        assert "n_samples" in result
        assert "metrics" in result
        assert "baseline_metrics" in result
        assert isinstance(result["n_samples"], int)
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["baseline_metrics"], dict)
        assert len(result["metrics"]) > 0
        assert len(result["baseline_metrics"]) > 0

    def test_different_seed_changes_metrics(self) -> None:
        """Different seeds should generally change the metrics (not guaranteed, but likely)."""
        r1 = run_experiment(seed=42, n_samples=256)
        r2 = run_experiment(seed=99, n_samples=256)
        # At least one metric should differ
        metrics_differ = r1["metrics"] != r2["metrics"]
        baseline_differ = r1["baseline_metrics"] != r2["baseline_metrics"]
        assert metrics_differ or baseline_differ
