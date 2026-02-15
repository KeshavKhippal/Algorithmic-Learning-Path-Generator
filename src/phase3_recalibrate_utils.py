"""
Phase 3 recalibration utilities.

Provides:
- Difficulty smoothing (neighbour-aware, α-weighted, iterative)
- Authority weighting (log-scaled resource count)
- Vectorised confidence recompute with configurable weights
- Percentile-based threshold derivation
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Difficulty smoothing
# =====================================================================


def smooth_difficulty(
    difficulties: np.ndarray,
    neighbor_ids_matrix: np.ndarray,
    ids: np.ndarray,
    alpha: float = 0.7,
    iterations: int = 1,
) -> np.ndarray:
    """Neighbor-aware difficulty smoothing.

    For each concept *i*, computes::

        d_smooth[i] = α * d[i] + (1 - α) * mean(d[neighbors_of_i])

    Invalid neighbors (id == -1) are excluded from the mean.

    Args:
        difficulties: ``(N,)`` original difficulty scores.
        neighbor_ids_matrix: ``(N, k)`` int64 neighbor IDs from FAISS.
        ids: ``(N,)`` int64 concept IDs (position → ID mapping).
        alpha: Weight for original difficulty (0..1).
        iterations: Number of smoothing passes.

    Returns:
        ``(N,)`` smoothed difficulty scores, clipped to [0, 1].
    """
    id_to_idx = {int(cid): idx for idx, cid in enumerate(ids)}
    d = difficulties.copy().astype(np.float64)

    for it in range(iterations):
        d_new = d.copy()
        for i in range(len(d)):
            neighbor_diffs = []
            for j in range(neighbor_ids_matrix.shape[1]):
                nid = int(neighbor_ids_matrix[i, j])
                if nid < 0 or nid == int(ids[i]):
                    continue
                idx = id_to_idx.get(nid)
                if idx is not None:
                    neighbor_diffs.append(d[idx])
            if neighbor_diffs:
                d_new[i] = alpha * d[i] + (1 - alpha) * np.mean(neighbor_diffs)
        d = d_new
        logger.debug("Smoothing iteration %d: mean=%.4f, std=%.4f", it + 1, d.mean(), d.std())

    # Clip and return
    d = np.clip(d, 0.0, 1.0).astype(np.float32)
    logger.info(
        "Difficulty smoothed (α=%.2f, iters=%d): mean=%.4f, std=%.4f, "
        "min=%.4f, max=%.4f",
        alpha, iterations, d.mean(), d.std(), d.min(), d.max(),
    )
    return d


# =====================================================================
# Authority weighting
# =====================================================================


def compute_authority(resource_counts: np.ndarray) -> np.ndarray:
    """Compute authority score per concept: 1 + log(resource_count).

    Args:
        resource_counts: ``(N,)`` integer resource counts.

    Returns:
        ``(N,)`` float32 authority scores (>= 1.0).
    """
    rc = np.maximum(resource_counts.astype(np.float64), 1.0)
    auth = 1.0 + np.log(rc)
    return auth.astype(np.float32)


def authority_adjusted_difficulty(
    difficulties: np.ndarray,
    resource_counts: np.ndarray,
    authority_weight: float = 0.15,
) -> np.ndarray:
    """Adjust difficulty by authority: concepts with higher authority
    get slightly lower difficulty (they're more "foundational").

    Formula::

        d_adj = d * (1 - authority_weight * norm_authority)

    where norm_authority is authority normalised to [0, 1].

    Args:
        difficulties: ``(N,)`` difficulty scores.
        resource_counts: ``(N,)`` resource counts.
        authority_weight: How much authority biases difficulty down (0..1).

    Returns:
        ``(N,)`` adjusted difficulty scores, clipped to [0, 1].
    """
    auth = compute_authority(resource_counts)
    if auth.max() > auth.min():
        norm_auth = (auth - auth.min()) / (auth.max() - auth.min())
    else:
        norm_auth = np.zeros_like(auth)

    d_adj = difficulties * (1.0 - authority_weight * norm_auth)
    d_adj = np.clip(d_adj, 0.0, 1.0).astype(np.float32)

    logger.info(
        "Authority-adjusted difficulty: mean=%.4f, std=%.4f",
        d_adj.mean(), d_adj.std(),
    )
    return d_adj


# =====================================================================
# Vectorised confidence computation
# =====================================================================


def compute_confidence_vectorised(
    similarities: np.ndarray,
    difficulty_gaps: np.ndarray,
    cooccurrence_bonuses: np.ndarray,
    w_sim: float = 0.60,
    w_gap: float = 0.30,
    w_cooc: float = 0.10,
) -> np.ndarray:
    """Compute confidence scores vectorised.

    Args:
        similarities: ``(M,)`` cosine similarities.
        difficulty_gaps: ``(M,)`` normalised difficulty gaps (clipped 0..1).
        cooccurrence_bonuses: ``(M,)`` co-occurrence values (0..1).
        w_sim, w_gap, w_cooc: Weights summing to 1.0.

    Returns:
        ``(M,)`` float32 confidence scores.
    """
    conf = (
        w_sim * similarities
        + w_gap * np.clip(difficulty_gaps, 0.0, 1.0)
        + w_cooc * cooccurrence_bonuses
    )
    return conf.astype(np.float32)


# =====================================================================
# Percentile threshold
# =====================================================================


def percentile_threshold(
    confidences: np.ndarray,
    percentile: float = 75.0,
) -> float:
    """Derive a confidence threshold from a percentile of the distribution.

    Args:
        confidences: All candidate confidence values.
        percentile: Which percentile to use (0..100).

    Returns:
        Threshold value.
    """
    if len(confidences) == 0:
        return 0.0
    return float(np.percentile(confidences, percentile))


# =====================================================================
# Candidate edge builder (vectorised, for sweeps)
# =====================================================================


def build_candidate_arrays(
    similarities_matrix: np.ndarray,
    neighbor_ids_matrix: np.ndarray,
    ids: np.ndarray,
    difficulties: np.ndarray,
    resource_sets: Dict[int, Set[int]],
    id_to_idx: Dict[int, int],
) -> Dict[str, np.ndarray]:
    """Flatten FAISS results into parallel arrays for vectorised sweeps.

    Returns dict with keys: source_ids, target_ids, sims, diff_gaps,
    cooc_bonuses, src_diffs, tgt_diffs. Each is a 1-D array of length M
    (total valid non-self neighbor pairs).
    """
    N, K = similarities_matrix.shape

    # Pre-allocate (upper bound = N*K)
    src_ids = np.empty(N * K, dtype=np.int64)
    tgt_ids = np.empty(N * K, dtype=np.int64)
    sims = np.empty(N * K, dtype=np.float32)
    diff_gaps = np.empty(N * K, dtype=np.float32)
    cooc_bonuses = np.empty(N * K, dtype=np.float32)
    src_diffs = np.empty(N * K, dtype=np.float32)
    tgt_diffs = np.empty(N * K, dtype=np.float32)

    m = 0
    for row in range(N):
        src_id = int(ids[row])
        src_diff = float(difficulties[id_to_idx[src_id]])
        src_res = resource_sets.get(src_id, set())

        for col in range(K):
            tgt_id = int(neighbor_ids_matrix[row, col])
            if tgt_id < 0 or tgt_id == src_id:
                continue
            tgt_idx = id_to_idx.get(tgt_id)
            if tgt_idx is None:
                continue

            tgt_diff = float(difficulties[tgt_idx])
            sim = float(similarities_matrix[row, col])

            tgt_res = resource_sets.get(tgt_id, set())
            max_rc = max(len(src_res), len(tgt_res), 1)
            cooc = len(src_res & tgt_res) / max_rc

            src_ids[m] = src_id
            tgt_ids[m] = tgt_id
            sims[m] = sim
            diff_gaps[m] = tgt_diff - src_diff
            cooc_bonuses[m] = cooc
            src_diffs[m] = src_diff
            tgt_diffs[m] = tgt_diff
            m += 1

    # Trim
    return {
        "source_ids": src_ids[:m],
        "target_ids": tgt_ids[:m],
        "sims": sims[:m],
        "diff_gaps": diff_gaps[:m],
        "cooc_bonuses": cooc_bonuses[:m],
        "src_diffs": src_diffs[:m],
        "tgt_diffs": tgt_diffs[:m],
    }


def apply_filters_and_count(
    arrays: Dict[str, np.ndarray],
    sim_min: float,
    conf_threshold: float,
    max_out: int,
    w_sim: float = 0.60,
    w_gap: float = 0.30,
    w_cooc: float = 0.10,
) -> Dict[str, Any]:
    """Apply filters to candidate arrays and return edge metrics.

    This is the core function used for sweeps — vectorised, no persistence.

    Returns dict with: edges_count, avg_out_degree, unique_sources, etc.
    """
    sims = arrays["sims"]
    diff_gaps = arrays["diff_gaps"]
    cooc = arrays["cooc_bonuses"]
    src_ids = arrays["source_ids"]
    tgt_ids = arrays["target_ids"]

    # 1. Similarity filter
    mask = sims >= sim_min

    # 2. Difficulty direction: gap must be positive
    mask &= diff_gaps > 0

    # 3. Compute confidence
    norm_gaps = np.clip(diff_gaps, 0.0, 1.0)
    conf = compute_confidence_vectorised(sims, norm_gaps, cooc, w_sim, w_gap, w_cooc)

    # 4. Confidence filter
    mask &= conf >= conf_threshold

    # Apply mask
    f_src = src_ids[mask]
    f_tgt = tgt_ids[mask]
    f_conf = conf[mask]

    if len(f_src) == 0:
        return {
            "edges_count": 0,
            "avg_out_degree": 0.0,
            "unique_sources": 0,
            "unique_targets": 0,
        }

    # 5. Max-out pruning: keep top max_out per source
    # Sort by (source, -confidence)
    order = np.lexsort((-f_conf, f_src))
    f_src = f_src[order]
    f_tgt = f_tgt[order]
    f_conf = f_conf[order]

    # Count per source, keep only first max_out
    keep_mask = np.ones(len(f_src), dtype=bool)
    current_src = -1
    count = 0
    for i in range(len(f_src)):
        if f_src[i] != current_src:
            current_src = f_src[i]
            count = 1
        else:
            count += 1
        if count > max_out:
            keep_mask[i] = False

    f_src = f_src[keep_mask]
    f_tgt = f_tgt[keep_mask]

    unique_sources = len(np.unique(f_src))
    unique_targets = len(np.unique(f_tgt))
    all_nodes = len(np.unique(np.concatenate([f_src, f_tgt])))

    return {
        "edges_count": int(len(f_src)),
        "avg_out_degree": round(len(f_src) / max(unique_sources, 1), 4),
        "unique_sources": int(unique_sources),
        "unique_targets": int(unique_targets),
        "nodes_in_graph": int(all_nodes),
    }
