"""
Phase 3 Structural Diagnostic — instruments the graph-building pipeline
without modifying it.  Produces 6 JSON reports in data/.

Usage::

    python scripts/phase3_diagnostic.py --db ./data/resources.db

Reads the same data as graph_builder.py, re-runs the FAISS query and
edge-scoring logic in read-only mode, then writes:
  - data/phase3_similarity_stats.json
  - data/phase3_difficulty_gap_stats.json
  - data/phase3_confidence_stats.json
  - data/phase3_isolated_analysis.json
  - data/phase3_degree_stats.json
  - data/phase3_rejection_stats.json
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db as phase1_db
from src.db_graph import (
    get_all_edges,
    load_canonical_concepts,
    load_canonical_embeddings,
    load_resource_sets,
    migrate_phase3,
)
from src.faiss_index import build_index, query_topk
from src.utils import setup_logging
from src.utils_graph import set_phase3_seed, timed

logger = logging.getLogger(__name__)


# =====================================================================
# Helpers
# =====================================================================

def _percentile(arr, p):
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, p))


def _pct_above(arr, threshold):
    if len(arr) == 0:
        return 0.0
    return round(float(np.sum(arr >= threshold) / len(arr) * 100), 4)


def _histogram(arr, bins=20):
    if len(arr) == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 6) for e in edges],
    }


def _save(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved → %s", path)


# =====================================================================
# Main diagnostic
# =====================================================================

def run_diagnostic(
    db_path: str = "./data/resources.db",
    top_k: int = 15,
    max_out: int = 8,
    sim_min: float = 0.50,
    conf_threshold: float = 0.55,
):
    set_phase3_seed(42)
    data_dir = os.path.dirname(os.path.abspath(db_path))

    migrate_phase3(db_path)
    conn = phase1_db.get_connection(db_path)

    try:
        # ---- Load data ----
        concepts = load_canonical_concepts(conn)
        concept_map: Dict[int, Dict[str, Any]] = {c["id"]: c for c in concepts}

        embeddings, ids, id_to_idx = load_canonical_embeddings(conn)
        resource_sets = load_resource_sets(conn)

        N, D = embeddings.shape
        logger.info("Diagnostic on N=%d concepts, D=%d dims.", N, D)

        # ---- FAISS query ----
        with timed("FAISS query"):
            index = build_index(embeddings, ids)
            similarities, neighbor_ids = query_topk(index, embeddings, k=top_k)

        # ================================================================
        # 1. SIMILARITY DISTRIBUTION (all top-k pairs before filtering)
        # ================================================================
        all_sims = []
        for row in range(N):
            src_id = int(ids[row])
            for col in range(similarities.shape[1]):
                tgt_id = int(neighbor_ids[row, col])
                if tgt_id < 0 or tgt_id == src_id:
                    continue
                all_sims.append(float(similarities[row, col]))

        all_sims_arr = np.array(all_sims, dtype=np.float64)
        sim_stats = {
            "total_pairs": len(all_sims),
            "min": round(float(all_sims_arr.min()), 6) if len(all_sims) else 0,
            "max": round(float(all_sims_arr.max()), 6) if len(all_sims) else 0,
            "mean": round(float(all_sims_arr.mean()), 6) if len(all_sims) else 0,
            "median": round(_percentile(all_sims_arr, 50), 6),
            "p25": round(_percentile(all_sims_arr, 25), 6),
            "p75": round(_percentile(all_sims_arr, 75), 6),
            "pct_gte_0.50": _pct_above(all_sims_arr, 0.50),
            "pct_gte_0.45": _pct_above(all_sims_arr, 0.45),
            "pct_gte_0.40": _pct_above(all_sims_arr, 0.40),
            "histogram": _histogram(all_sims_arr, bins=25),
        }
        _save(sim_stats, os.path.join(data_dir, "phase3_similarity_stats.json"))

        # ================================================================
        # 2+3+6. Walk all pairs: difficulty gaps, confidence, rejections
        # ================================================================
        diff_gaps_all = []         # all neighbor pairs
        confidences_valid = []     # difficulty-ordered + sim-passing
        rejection_counts = {
            "self_loop": 0,
            "similarity_below_min": 0,
            "difficulty_wrong_direction": 0,
            "confidence_below_threshold": 0,
            "max_out_pruned": 0,
        }

        # Edges per source before max_out pruning
        edges_per_source: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        total_examined = 0

        for row in range(N):
            src_id = int(ids[row])
            src = concept_map.get(src_id)
            if src is None:
                continue
            src_diff = src["difficulty_score"] or 0.0

            for col in range(similarities.shape[1]):
                tgt_id = int(neighbor_ids[row, col])
                sim = float(similarities[row, col])

                if tgt_id < 0:
                    continue
                if tgt_id == src_id:
                    rejection_counts["self_loop"] += 1
                    continue

                total_examined += 1
                tgt = concept_map.get(tgt_id)
                if tgt is None:
                    continue
                tgt_diff = tgt["difficulty_score"] or 0.0

                diff_gap = tgt_diff - src_diff
                diff_gaps_all.append(diff_gap)

                # Similarity filter
                if sim < sim_min:
                    rejection_counts["similarity_below_min"] += 1
                    continue

                # Difficulty direction
                if src_diff >= tgt_diff:
                    rejection_counts["difficulty_wrong_direction"] += 1
                    continue

                # Compute confidence
                src_res = resource_sets.get(src_id, set())
                tgt_res = resource_sets.get(tgt_id, set())
                max_rc = max(len(src_res), len(tgt_res), 1)
                cooc = len(src_res & tgt_res) / max_rc
                diff_gap_norm = min((tgt_diff - src_diff) / 1.0, 1.0)
                confidence = 0.60 * sim + 0.30 * diff_gap_norm + 0.10 * cooc
                confidences_valid.append(confidence)

                if confidence < conf_threshold:
                    rejection_counts["confidence_below_threshold"] += 1
                    continue

                edges_per_source[src_id].append({
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "similarity": sim,
                    "confidence": confidence,
                })

        # Max-out pruning
        final_edge_count = 0
        for src_id, edges in edges_per_source.items():
            edges.sort(key=lambda e: e["confidence"], reverse=True)
            kept = edges[:max_out]
            pruned = len(edges) - len(kept)
            rejection_counts["max_out_pruned"] += pruned
            final_edge_count += len(kept)

        total_pass_sim = total_examined - rejection_counts["similarity_below_min"]
        total_pass_diff = total_pass_sim - rejection_counts["difficulty_wrong_direction"]
        total_pass_conf = total_pass_diff - rejection_counts["confidence_below_threshold"]

        # ---- 2. Difficulty gap stats ----
        diff_gaps_arr = np.array(diff_gaps_all, dtype=np.float64)
        diff_stats = {
            "total_pairs": len(diff_gaps_all),
            "mean": round(float(diff_gaps_arr.mean()), 6) if len(diff_gaps_all) else 0,
            "median": round(_percentile(diff_gaps_arr, 50), 6),
            "pct_positive": _pct_above(diff_gaps_arr, 1e-9),
            "pct_negative": round(float(np.sum(diff_gaps_arr < -1e-9) / max(len(diff_gaps_all), 1) * 100), 4),
            "pct_zero": round(float(np.sum(np.abs(diff_gaps_arr) <= 1e-9) / max(len(diff_gaps_all), 1) * 100), 4),
            "histogram": _histogram(diff_gaps_arr, bins=25),
        }
        _save(diff_stats, os.path.join(data_dir, "phase3_difficulty_gap_stats.json"))

        # ---- 3. Confidence stats ----
        conf_arr = np.array(confidences_valid, dtype=np.float64)
        conf_stats = {
            "total_valid_pairs": len(confidences_valid),
            "mean": round(float(conf_arr.mean()), 6) if len(confidences_valid) else 0,
            "median": round(_percentile(conf_arr, 50), 6),
            "pct_gte_0.55": _pct_above(conf_arr, 0.55),
            "pct_gte_0.50": _pct_above(conf_arr, 0.50),
            "pct_gte_0.45": _pct_above(conf_arr, 0.45),
            "histogram": _histogram(conf_arr, bins=25),
        }
        _save(conf_stats, os.path.join(data_dir, "phase3_confidence_stats.json"))

        # ---- 6. Rejection stats ----
        rejection_stats = {
            "total_neighbor_pairs_examined": total_examined,
            "total_passing_similarity_filter": total_pass_sim,
            "total_passing_difficulty_rule": total_pass_diff,
            "total_passing_confidence": total_pass_conf,
            "final_edges_after_max_out": final_edge_count,
            "rejection_breakdown": rejection_counts,
        }
        _save(rejection_stats, os.path.join(data_dir, "phase3_rejection_stats.json"))

        # ================================================================
        # 4. ISOLATED NODE ANALYSIS
        # ================================================================
        existing_edges = get_all_edges(conn)
        nodes_in_graph: Set[int] = set()
        in_degree: Counter = Counter()
        out_degree: Counter = Counter()

        for e in existing_edges:
            src = e["source_concept_id"]
            tgt = e["target_concept_id"]
            nodes_in_graph.add(src)
            nodes_in_graph.add(tgt)
            out_degree[src] += 1
            in_degree[tgt] += 1

        all_concept_ids = {c["id"] for c in concepts}
        isolated_ids = all_concept_ids - nodes_in_graph

        isolated_concepts = [concept_map[cid] for cid in isolated_ids if cid in concept_map]
        iso_diffs = [c["difficulty_score"] or 0.0 for c in isolated_concepts]
        iso_rcs = [c["resource_count"] or 0 for c in isolated_concepts]

        iso_diffs_arr = np.array(iso_diffs, dtype=np.float64)
        iso_rcs_arr = np.array(iso_rcs, dtype=np.float64)

        # Sort for top-20 lists
        iso_by_freq = sorted(isolated_concepts, key=lambda c: c["resource_count"] or 0, reverse=True)
        iso_by_low_diff = sorted(isolated_concepts, key=lambda c: c["difficulty_score"] or 0.0)
        iso_by_high_diff = sorted(isolated_concepts, key=lambda c: c["difficulty_score"] or 0.0, reverse=True)

        isolated_analysis = {
            "total_isolated": len(isolated_ids),
            "avg_difficulty": round(float(iso_diffs_arr.mean()), 6) if len(iso_diffs) else 0,
            "median_difficulty": round(_percentile(iso_diffs_arr, 50), 6),
            "avg_resource_count": round(float(iso_rcs_arr.mean()), 4) if len(iso_rcs) else 0,
            "median_resource_count": round(_percentile(iso_rcs_arr, 50), 4),
            "top_20_most_frequent": [
                {"concept": c["canonical_concept"], "resource_count": c["resource_count"],
                 "difficulty": c["difficulty_score"]}
                for c in iso_by_freq[:20]
            ],
            "top_20_lowest_difficulty": [
                {"concept": c["canonical_concept"], "difficulty": c["difficulty_score"],
                 "resource_count": c["resource_count"]}
                for c in iso_by_low_diff[:20]
            ],
            "top_20_highest_difficulty": [
                {"concept": c["canonical_concept"], "difficulty": c["difficulty_score"],
                 "resource_count": c["resource_count"]}
                for c in iso_by_high_diff[:20]
            ],
        }
        _save(isolated_analysis, os.path.join(data_dir, "phase3_isolated_analysis.json"))

        # ================================================================
        # 5. DEGREE DISTRIBUTION
        # ================================================================
        all_in = [in_degree.get(cid, 0) for cid in nodes_in_graph]
        all_out = [out_degree.get(cid, 0) for cid in nodes_in_graph]

        in_dist = Counter(all_in)
        out_dist = Counter(all_out)

        # Top 20 by degree
        top_in = in_degree.most_common(20)
        top_out = out_degree.most_common(20)

        degree_stats = {
            "in_degree_distribution": dict(sorted(in_dist.items())),
            "out_degree_distribution": dict(sorted(out_dist.items())),
            "top_20_by_in_degree": [
                {"concept_id": cid, "concept": concept_map.get(cid, {}).get("canonical_concept", "?"),
                 "in_degree": deg, "difficulty": concept_map.get(cid, {}).get("difficulty_score")}
                for cid, deg in top_in
            ],
            "top_20_by_out_degree": [
                {"concept_id": cid, "concept": concept_map.get(cid, {}).get("canonical_concept", "?"),
                 "out_degree": deg, "difficulty": concept_map.get(cid, {}).get("difficulty_score")}
                for cid, deg in top_out
            ],
        }
        _save(degree_stats, os.path.join(data_dir, "phase3_degree_stats.json"))

        # ================================================================
        # 7. PRINT SUMMARY
        # ================================================================
        print("\n" + "=" * 60)
        print("PHASE 3 STRUCTURAL DIAGNOSTIC")
        print("=" * 60)
        print(f"  Concepts (N):                  {N}")
        print(f"  Embedding dim (D):             {D}")
        print(f"  Top-k:                         {top_k}")
        print()
        print(f"  Total neighbor pairs examined: {total_examined:,}")
        print(f"  Passing similarity (≥{sim_min}):  {total_pass_sim:,}")
        print(f"  Passing difficulty rule:       {total_pass_diff:,}")
        print(f"  Passing confidence (≥{conf_threshold}):  {total_pass_conf:,}")
        print(f"  After max_out={max_out} pruning:      {final_edge_count:,}")
        print(f"  Final edges in DB:             {len(existing_edges):,}")
        print()
        print("  -- Rejection Breakdown --")
        for reason, count in rejection_counts.items():
            print(f"    {reason:35s} {count:>8,}")
        print()
        print(f"  -- Similarity Stats --")
        print(f"    min={sim_stats['min']:.4f}  max={sim_stats['max']:.4f}  "
              f"mean={sim_stats['mean']:.4f}  median={sim_stats['median']:.4f}")
        print(f"    p25={sim_stats['p25']:.4f}  p75={sim_stats['p75']:.4f}")
        print(f"    ≥0.50: {sim_stats['pct_gte_0.50']:.1f}%   "
              f"≥0.45: {sim_stats['pct_gte_0.45']:.1f}%   "
              f"≥0.40: {sim_stats['pct_gte_0.40']:.1f}%")
        print()
        print(f"  -- Difficulty Gap Stats --")
        print(f"    mean={diff_stats['mean']:.4f}  median={diff_stats['median']:.4f}")
        print(f"    positive: {diff_stats['pct_positive']:.1f}%  "
              f"negative: {diff_stats['pct_negative']:.1f}%  "
              f"zero: {diff_stats['pct_zero']:.1f}%")
        print()
        print(f"  -- Confidence Stats --")
        print(f"    mean={conf_stats['mean']:.4f}  median={conf_stats['median']:.4f}")
        print(f"    ≥0.55: {conf_stats['pct_gte_0.55']:.1f}%  "
              f"≥0.50: {conf_stats['pct_gte_0.50']:.1f}%  "
              f"≥0.45: {conf_stats['pct_gte_0.45']:.1f}%")
        print()
        print(f"  -- Isolated Nodes --")
        print(f"    count: {isolated_analysis['total_isolated']:,} / {len(concepts):,}")
        print(f"    avg difficulty: {isolated_analysis['avg_difficulty']:.4f}")
        print(f"    avg resource_count: {isolated_analysis['avg_resource_count']:.2f}")
        print("=" * 60)

    finally:
        conn.close()


# =====================================================================
# CLI
# =====================================================================

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 3 structural diagnostic")
    parser.add_argument("--db", default="./data/resources.db")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-out", type=int, default=8)
    parser.add_argument("--sim-min", type=float, default=0.50)
    parser.add_argument("--conf-threshold", type=float, default=0.55)
    args = parser.parse_args()

    run_diagnostic(
        db_path=args.db,
        top_k=args.top_k,
        max_out=args.max_out,
        sim_min=args.sim_min,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
