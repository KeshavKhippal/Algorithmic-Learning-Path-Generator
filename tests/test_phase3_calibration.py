"""
Tests for Phase 3 calibration: difficulty smoothing, authority weighting,
confidence recompute, and sensitivity sweep on sample data.

All tests are deterministic with seed 42.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.phase3_recalibrate_utils import (
    apply_filters_and_count,
    authority_adjusted_difficulty,
    build_candidate_arrays,
    compute_authority,
    compute_confidence_vectorised,
    percentile_threshold,
    smooth_difficulty,
)
from src.utils_graph import set_phase3_seed


# =========================================================================
# Helpers
# =========================================================================

def _make_embeddings(n, dim=384, seed=42):
    np.random.seed(seed)
    emb = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def _make_similar_pair(dim=384, seed=42):
    np.random.seed(seed)
    base = np.random.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    noise = np.random.randn(dim).astype(np.float32) * 0.01
    other = base + noise
    other /= np.linalg.norm(other)
    return base, other


# =========================================================================
# Test: Difficulty Smoothing
# =========================================================================


class TestDifficultySmoothing:
    """Tests for neighbour-aware difficulty smoothing."""

    def test_deterministic(self):
        """Same inputs → same outputs with seed."""
        set_phase3_seed(42)
        diffs = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        ids = np.arange(1, 6, dtype=np.int64)
        # neighbors: each concept has the next one
        neighbors = np.array([
            [1, 2, -1], [2, 3, -1], [3, 4, -1],
            [4, 5, -1], [5, 1, -1],
        ], dtype=np.int64)

        r1 = smooth_difficulty(diffs, neighbors, ids, alpha=0.7, iterations=1)
        r2 = smooth_difficulty(diffs, neighbors, ids, alpha=0.7, iterations=1)
        np.testing.assert_array_equal(r1, r2)

    def test_output_range(self):
        """Smoothed difficulties remain in [0, 1]."""
        diffs = np.array([0.0, 0.5, 1.0, 0.3, 0.8], dtype=np.float32)
        ids = np.arange(1, 6, dtype=np.int64)
        neighbors = np.array([
            [1, 2, 3], [2, 1, 3], [3, 4, 5],
            [4, 5, 1], [5, 1, 2],
        ], dtype=np.int64)

        result = smooth_difficulty(diffs, neighbors, ids, alpha=0.7, iterations=2)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_alpha_1_unchanged(self):
        """α=1 → output equals input (no smoothing)."""
        diffs = np.array([0.2, 0.8, 0.5], dtype=np.float32)
        ids = np.array([10, 20, 30], dtype=np.int64)
        neighbors = np.array([[10, 20, -1], [20, 30, -1], [30, 10, -1]], dtype=np.int64)

        result = smooth_difficulty(diffs, neighbors, ids, alpha=1.0, iterations=3)
        np.testing.assert_allclose(result, diffs, atol=1e-6)

    def test_smoothing_reduces_variance(self):
        """Smoothing should reduce difficulty variance."""
        diffs = np.array([0.05, 0.95, 0.1, 0.9, 0.5], dtype=np.float32)
        ids = np.arange(1, 6, dtype=np.int64)
        # All connected to each other
        neighbors = np.array([
            [1, 2, 3], [2, 1, 4], [3, 4, 5],
            [4, 3, 5], [5, 1, 2],
        ], dtype=np.int64)

        smoothed = smooth_difficulty(diffs, neighbors, ids, alpha=0.5, iterations=3)
        assert smoothed.std() < diffs.std()

    def test_monotonicity_preserved_locally(self):
        """If concept A has much lower difficulty than all its neighbors,
        it should not end up dramatically higher after smoothing."""
        diffs = np.array([0.1, 0.8, 0.9, 0.85, 0.7], dtype=np.float32)
        ids = np.arange(1, 6, dtype=np.int64)
        neighbors = np.array([
            [1, 2, 3], [2, 3, 4], [3, 4, 5],
            [4, 5, 2], [5, 3, 4],
        ], dtype=np.int64)

        smoothed = smooth_difficulty(diffs, neighbors, ids, alpha=0.7, iterations=2)
        # Concept 0 (diff=0.1) should still be lower than concept 2 (diff=0.9)
        assert smoothed[0] < smoothed[2]


# =========================================================================
# Test: Authority Weighting
# =========================================================================


class TestAuthorityWeighting:
    """Tests for authority score computation."""

    def test_authority_monotonic(self):
        """Higher resource_count → higher authority."""
        rc = np.array([1, 5, 10, 50], dtype=np.int32)
        auth = compute_authority(rc)
        for i in range(len(auth) - 1):
            assert auth[i] < auth[i + 1]

    def test_authority_min_one(self):
        """Authority is always >= 1.0."""
        rc = np.array([0, 1, 2], dtype=np.int32)
        auth = compute_authority(rc)
        assert auth.min() >= 1.0

    def test_authority_adjusted_lowers_high_resource(self):
        """Concepts with higher resource_count should get lower adjusted difficulty."""
        diffs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        rc = np.array([1, 10, 100], dtype=np.int32)

        adjusted = authority_adjusted_difficulty(diffs, rc, authority_weight=0.15)
        # Higher resource_count → lower difficulty
        assert adjusted[0] > adjusted[1] > adjusted[2]


# =========================================================================
# Test: Confidence Computation
# =========================================================================


class TestConfidenceComputation:
    """Tests for vectorised confidence and percentile threshold."""

    def test_confidence_weights_sum(self):
        """Confidence at max inputs should equal sum of weights."""
        sims = np.array([1.0], dtype=np.float32)
        gaps = np.array([1.0], dtype=np.float32)
        cooc = np.array([1.0], dtype=np.float32)
        conf = compute_confidence_vectorised(sims, gaps, cooc, 0.6, 0.3, 0.1)
        np.testing.assert_allclose(conf, [1.0], atol=1e-5)

    def test_confidence_higher_sim_higher_score(self):
        """Higher similarity → higher confidence."""
        sims = np.array([0.3, 0.7, 0.9], dtype=np.float32)
        gaps = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        cooc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        conf = compute_confidence_vectorised(sims, gaps, cooc, 0.7, 0.2, 0.1)
        assert conf[0] < conf[1] < conf[2]

    def test_percentile_threshold_basic(self):
        """Percentile threshold on uniform data."""
        conf = np.linspace(0.0, 1.0, 101).astype(np.float32)
        assert abs(percentile_threshold(conf, 50) - 0.5) < 0.02
        assert abs(percentile_threshold(conf, 75) - 0.75) < 0.02

    def test_percentile_threshold_empty(self):
        """Empty array returns 0."""
        assert percentile_threshold(np.array([], dtype=np.float32), 75) == 0.0


# =========================================================================
# Test: Candidate Arrays + Filter
# =========================================================================


class TestCandidateArraysAndFilter:
    """Tests for build_candidate_arrays and apply_filters_and_count."""

    def _setup(self):
        """Create a small setup with 5 concepts."""
        set_phase3_seed(42)
        emb = _make_embeddings(5)
        ids = np.arange(1, 6, dtype=np.int64)
        id_to_idx = {i: i - 1 for i in range(1, 6)}
        diffs = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        resource_sets = {i: {1, 2} for i in range(1, 6)}

        from src.faiss_index import build_index, query_topk
        index = build_index(emb, ids)
        sims, neighbor_ids = query_topk(index, emb, k=5)

        arrays = build_candidate_arrays(
            sims, neighbor_ids, ids, diffs, resource_sets, id_to_idx,
        )
        return arrays, ids, diffs

    def test_arrays_no_self_loops(self):
        """Candidate arrays should not contain self-loops."""
        arrays, _, _ = self._setup()
        assert not np.any(arrays["source_ids"] == arrays["target_ids"])

    def test_filter_produces_edges(self):
        """With loose thresholds, should produce some edges."""
        arrays, _, _ = self._setup()
        result = apply_filters_and_count(
            arrays, sim_min=0.0, conf_threshold=0.0, max_out=10,
        )
        assert result["edges_count"] > 0

    def test_filter_strict_reduces_edges(self):
        """Strict thresholds produce fewer edges than loose."""
        arrays, _, _ = self._setup()
        loose = apply_filters_and_count(
            arrays, sim_min=0.0, conf_threshold=0.0, max_out=10,
        )
        strict = apply_filters_and_count(
            arrays, sim_min=0.50, conf_threshold=0.50, max_out=10,
        )
        assert strict["edges_count"] <= loose["edges_count"]

    def test_max_out_caps_edges(self):
        """max_out=1 should cap each source to 1 edge."""
        arrays, _, _ = self._setup()
        result = apply_filters_and_count(
            arrays, sim_min=0.0, conf_threshold=0.0, max_out=1,
        )
        # Unique sources should equal edges_count (1 edge per source)
        assert result["edges_count"] == result["unique_sources"]

    def test_difficulty_direction_enforced_in_filter(self):
        """Only positive difficulty gaps should produce edges."""
        arrays, _, _ = self._setup()
        result = apply_filters_and_count(
            arrays, sim_min=0.0, conf_threshold=0.0, max_out=10,
        )
        # All positive difficulty gaps after filtering
        mask = arrays["diff_gaps"] > 0
        assert result["edges_count"] <= int(mask.sum())


# =========================================================================
# Test: Determinism
# =========================================================================


class TestCalibrationDeterminism:
    """Tests that calibration functions produce deterministic output."""

    def test_smoothing_deterministic_across_calls(self):
        """Two calls with same seed produce identical smoothed difficulties."""
        diffs = np.array([0.1, 0.4, 0.7, 0.2, 0.9], dtype=np.float32)
        ids = np.arange(1, 6, dtype=np.int64)
        neighbors = np.array([
            [1, 2, 3], [2, 3, 4], [3, 4, 5],
            [4, 5, 1], [5, 1, 2],
        ], dtype=np.int64)

        set_phase3_seed(42)
        r1 = smooth_difficulty(diffs, neighbors, ids, alpha=0.7, iterations=2)
        set_phase3_seed(42)
        r2 = smooth_difficulty(diffs, neighbors, ids, alpha=0.7, iterations=2)
        np.testing.assert_array_equal(r1, r2)

    def test_authority_deterministic(self):
        """Authority computation is deterministic."""
        rc = np.array([1, 3, 10, 25], dtype=np.int32)
        a1 = compute_authority(rc)
        a2 = compute_authority(rc)
        np.testing.assert_array_equal(a1, a2)
