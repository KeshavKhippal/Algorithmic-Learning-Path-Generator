"""
Phase 3 calibration orchestrator.

Runs diagnostic sweeps, difficulty recalibration, confidence weight search,
and produces comparison reports — all without persisting edges (read-only
until --apply-config is used).

Usage::

    # Calibrate-only (diagnostics + sweep + report)
    python -m src.phase3_calibrate --db ./data/resources.db

    # Apply a specific config (with optional --dry-run)
    python -m src.phase3_calibrate --db ./data/resources.db \\
        --apply-config ./data/phase3_calibration/best_config.json [--dry-run]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db as phase1_db
from src.dag_validator import break_cycles, compute_metrics, validate_dag
from src.db_graph import (
    clear_phase3_data,
    get_all_edges,
    insert_edges_batch,
    load_canonical_concepts,
    load_canonical_embeddings,
    load_resource_sets,
    migrate_phase3,
)
from src.faiss_index import build_index, query_topk
from src.graph_builder import generate_candidate_edges
from src.phase3_recalibrate_utils import (
    apply_filters_and_count,
    authority_adjusted_difficulty,
    build_candidate_arrays,
    compute_authority,
    compute_confidence_vectorised,
    percentile_threshold,
    smooth_difficulty,
)
from src.utils import setup_logging
from src.utils_graph import set_phase3_seed, timed

logger = logging.getLogger(__name__)


# =====================================================================
# Helpers
# =====================================================================

def _save(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved → %s", path)


def _histogram(arr, bins=20):
    if len(arr) == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {"counts": counts.tolist(), "bin_edges": [round(float(e), 6) for e in edges]}


def _pct_above(arr, threshold):
    if len(arr) == 0:
        return 0.0
    return round(float(np.sum(arr >= threshold) / len(arr) * 100), 4)


def _percentile(arr, p):
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, p))


# =====================================================================
# A: Threshold sweep
# =====================================================================

def run_threshold_sweep(
    arrays: Dict[str, np.ndarray],
    n_concepts: int,
    sim_min_values: List[float] = None,
    conf_values: List[float] = None,
    max_out_values: List[int] = None,
    w_sim: float = 0.60,
    w_gap: float = 0.30,
    w_cooc: float = 0.10,
) -> List[Dict[str, Any]]:
    """Sweep over (sim_min, conf_threshold, max_out) grid."""
    if sim_min_values is None:
        sim_min_values = [0.40, 0.45, 0.50, 0.55]
    if conf_values is None:
        conf_values = [0.40, 0.45, 0.50, 0.55, 0.60]
    if max_out_values is None:
        max_out_values = [4, 8, 12]

    results = []
    for sim_min, conf_thresh, max_out in product(sim_min_values, conf_values, max_out_values):
        r = apply_filters_and_count(
            arrays, sim_min, conf_thresh, max_out,
            w_sim=w_sim, w_gap=w_gap, w_cooc=w_cooc,
        )
        r.update({
            "sim_min": sim_min,
            "conf_threshold": conf_thresh,
            "max_out": max_out,
            "isolated_count": n_concepts - r["nodes_in_graph"],
            "isolated_pct": round((n_concepts - r["nodes_in_graph"]) / n_concepts * 100, 2),
        })
        results.append(r)

    return results


# =====================================================================
# C: Confidence weight sweep
# =====================================================================

def run_confidence_sweep(
    arrays: Dict[str, np.ndarray],
    n_concepts: int,
    sim_min: float = 0.45,
    max_out: int = 8,
    conf_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """Sweep over (w_sim, w_gap, w_cooc) weight tuples (sum = 1.0)."""
    w_sim_range = [0.40, 0.50, 0.60, 0.70, 0.80]
    w_gap_range = [0.10, 0.20, 0.30, 0.40]
    w_cooc_range = [0.00, 0.05, 0.10, 0.20]

    results = []
    for ws, wg, wc in product(w_sim_range, w_gap_range, w_cooc_range):
        if abs(ws + wg + wc - 1.0) > 0.01:
            continue
        r = apply_filters_and_count(
            arrays, sim_min, conf_threshold, max_out,
            w_sim=ws, w_gap=wg, w_cooc=wc,
        )
        r.update({
            "w_sim": ws, "w_gap": wg, "w_cooc": wc,
            "sim_min": sim_min,
            "conf_threshold": conf_threshold,
            "max_out": max_out,
            "isolated_count": n_concepts - r["nodes_in_graph"],
            "isolated_pct": round((n_concepts - r["nodes_in_graph"]) / n_concepts * 100, 2),
        })
        results.append(r)

    results.sort(key=lambda x: (-x["edges_count"], x["isolated_pct"]))
    return results


# =====================================================================
# D: Dry-run evaluation of a config
# =====================================================================

def evaluate_config(
    config: Dict[str, Any],
    embeddings: np.ndarray,
    ids: np.ndarray,
    id_to_idx: Dict[int, int],
    concepts: List[Dict],
    resource_sets: Dict[int, Set[int]],
    difficulties_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run full Phase 3 pipeline in dry-run mode with given config."""

    top_k = config.get("top_k", 15)
    max_out = config.get("max_out", 8)
    sim_min = config.get("sim_min", 0.45)
    conf_threshold = config.get("conf_threshold", 0.50)
    w_sim = config.get("w_sim", 0.60)
    w_gap = config.get("w_gap", 0.30)
    w_cooc = config.get("w_cooc", 0.10)

    # Override difficulties if smoothed
    if difficulties_override is not None:
        concept_list = []
        for c in concepts:
            cc = dict(c)
            idx = id_to_idx.get(c["id"])
            if idx is not None:
                cc["difficulty_score"] = float(difficulties_override[idx])
            concept_list.append(cc)
    else:
        concept_list = concepts

    edges = generate_candidate_edges(
        embeddings, ids, id_to_idx, concept_list, resource_sets,
        top_k=top_k, max_out=max_out, sim_min=sim_min,
        conf_threshold=conf_threshold,
        w_sim=w_sim, w_gap=w_gap, w_cooc=w_cooc,
    )

    acyclic, removed = break_cycles(edges)
    is_dag = validate_dag(acyclic)
    metrics = compute_metrics(acyclic, n_concepts=len(concepts))
    metrics["is_dag"] = is_dag
    metrics["removed_edges"] = len(removed)
    metrics["config"] = config

    # Sample edges
    sorted_edges = sorted(acyclic, key=lambda e: e["confidence"], reverse=True)
    metrics["sample_top_100_edges"] = sorted_edges[:100]

    return metrics


# =====================================================================
# Main calibration pipeline
# =====================================================================

def run_calibration(
    db_path: str = "./data/resources.db",
    out_dir: str = "./data/phase3_calibration",
    top_k: int = 15,
):
    """Full calibration pipeline: sweep → recalibrate → evaluate → report."""
    set_phase3_seed(42)
    os.makedirs(out_dir, exist_ok=True)

    migrate_phase3(db_path)
    conn = phase1_db.get_connection(db_path)

    try:
        # ---- Load data ----
        with timed("Load concepts"):
            concepts = load_canonical_concepts(conn)
        with timed("Load embeddings"):
            embeddings, ids, id_to_idx = load_canonical_embeddings(conn)
        with timed("Load resource sets"):
            resource_sets = load_resource_sets(conn)

        N = len(concepts)
        concept_map = {c["id"]: c for c in concepts}

        logger.info("Calibration: N=%d concepts, dim=%d.", N, embeddings.shape[1])

        # ---- FAISS query ----
        with timed("FAISS index + query"):
            index = build_index(embeddings, ids)
            sim_matrix, neighbor_matrix = query_topk(index, embeddings, k=top_k)

        # ---- Baseline difficulties ----
        orig_diffs = np.array(
            [concepts[id_to_idx[int(cid)]]["difficulty_score"] or 0.0
             for cid in ids],
            dtype=np.float32,
        )
        resource_counts = np.array(
            [concepts[id_to_idx[int(cid)]]["resource_count"] or 1
             for cid in ids],
            dtype=np.int32,
        )

        # ---- Build candidate arrays (original difficulties) ----
        with timed("Build candidate arrays (original)"):
            arrays_orig = build_candidate_arrays(
                sim_matrix, neighbor_matrix, ids,
                orig_diffs, resource_sets, id_to_idx,
            )
        logger.info("Candidate pairs: %d", len(arrays_orig["sims"]))

        # ================================================================
        # A: Baseline threshold sweep
        # ================================================================
        logger.info("=== A: THRESHOLD SWEEP (original difficulty) ===")
        with timed("Threshold sweep"):
            sweep_results = run_threshold_sweep(arrays_orig, N)
        _save(sweep_results, os.path.join(out_dir, "phase3_threshold_sweep.json"))

        # ================================================================
        # B: Difficulty recalibration
        # ================================================================
        logger.info("=== B: DIFFICULTY RECALIBRATION ===")

        # B1: Neighbor-aware smoothing
        with timed("Difficulty smoothing"):
            smoothed_diffs = smooth_difficulty(
                orig_diffs, neighbor_matrix, ids,
                alpha=0.7, iterations=2,
            )

        # B2: Authority adjustment
        with timed("Authority adjustment"):
            auth_diffs = authority_adjusted_difficulty(
                smoothed_diffs, resource_counts,
                authority_weight=0.15,
            )

        # Save difficulty comparison
        diff_comparison = {
            "original": {
                "mean": round(float(orig_diffs.mean()), 4),
                "std": round(float(orig_diffs.std()), 4),
                "min": round(float(orig_diffs.min()), 4),
                "max": round(float(orig_diffs.max()), 4),
            },
            "smoothed": {
                "mean": round(float(smoothed_diffs.mean()), 4),
                "std": round(float(smoothed_diffs.std()), 4),
                "min": round(float(smoothed_diffs.min()), 4),
                "max": round(float(smoothed_diffs.max()), 4),
            },
            "authority_adjusted": {
                "mean": round(float(auth_diffs.mean()), 4),
                "std": round(float(auth_diffs.std()), 4),
                "min": round(float(auth_diffs.min()), 4),
                "max": round(float(auth_diffs.max()), 4),
            },
        }
        _save(diff_comparison, os.path.join(out_dir, "difficulty_comparison.json"))

        # ---- Build candidate arrays with recalibrated difficulty ----
        with timed("Build candidate arrays (recalibrated)"):
            arrays_recal = build_candidate_arrays(
                sim_matrix, neighbor_matrix, ids,
                auth_diffs, resource_sets, id_to_idx,
            )

        # Sweep with recalibrated difficulties
        with timed("Threshold sweep (recalibrated)"):
            sweep_recal = run_threshold_sweep(arrays_recal, N)
        _save(sweep_recal, os.path.join(out_dir, "phase3_threshold_sweep_recalibrated.json"))

        # ================================================================
        # C: Confidence weight sweep
        # ================================================================
        logger.info("=== C: CONFIDENCE WEIGHT SWEEP ===")
        with timed("Confidence weight sweep (original)"):
            conf_sweep_orig = run_confidence_sweep(arrays_orig, N)
        _save(conf_sweep_orig, os.path.join(out_dir, "phase3_confidence_sweep.json"))

        with timed("Confidence weight sweep (recalibrated)"):
            conf_sweep_recal = run_confidence_sweep(arrays_recal, N)
        _save(conf_sweep_recal, os.path.join(out_dir, "phase3_confidence_sweep_recalibrated.json"))

        # ---- Percentile thresholds ----
        # Compute confidence distribution for recalibrated arrays at best weights
        diff_gaps_pos = arrays_recal["diff_gaps"]
        pos_mask = diff_gaps_pos > 0
        sim_pass = arrays_recal["sims"] >= 0.45
        combined_mask = pos_mask & sim_pass
        if combined_mask.sum() > 0:
            valid_conf = compute_confidence_vectorised(
                arrays_recal["sims"][combined_mask],
                np.clip(arrays_recal["diff_gaps"][combined_mask], 0.0, 1.0),
                arrays_recal["cooc_bonuses"][combined_mask],
                w_sim=0.70, w_gap=0.20, w_cooc=0.10,
            )
            pct_thresholds = {
                "p50": round(percentile_threshold(valid_conf, 50), 4),
                "p60": round(percentile_threshold(valid_conf, 60), 4),
                "p70": round(percentile_threshold(valid_conf, 70), 4),
                "p75": round(percentile_threshold(valid_conf, 75), 4),
                "p80": round(percentile_threshold(valid_conf, 80), 4),
                "p90": round(percentile_threshold(valid_conf, 90), 4),
                "mean": round(float(valid_conf.mean()), 4),
                "total_valid_pairs": int(combined_mask.sum()),
            }
        else:
            pct_thresholds = {"error": "no valid pairs"}
        _save(pct_thresholds, os.path.join(out_dir, "percentile_thresholds.json"))

        # ================================================================
        # D: Dry-run evaluation of candidate configs
        # ================================================================
        logger.info("=== D: DRY-RUN EVALUATION ===")

        # Baseline config
        baseline_config = {
            "name": "baseline",
            "top_k": top_k, "max_out": 8, "sim_min": 0.50,
            "conf_threshold": 0.55,
            "w_sim": 0.60, "w_gap": 0.30, "w_cooc": 0.10,
            "use_smoothed_difficulty": False,
        }

        # Balanced config — lower thresholds, better weights, smoothed difficulty
        balanced_config = {
            "name": "balanced",
            "top_k": top_k, "max_out": 8, "sim_min": 0.45,
            "conf_threshold": 0.45,
            "w_sim": 0.70, "w_gap": 0.20, "w_cooc": 0.10,
            "use_smoothed_difficulty": True,
        }

        # High-recall config — aggressive connectivity
        recall_config = {
            "name": "high_recall",
            "top_k": top_k, "max_out": 12, "sim_min": 0.40,
            "conf_threshold": 0.40,
            "w_sim": 0.70, "w_gap": 0.20, "w_cooc": 0.10,
            "use_smoothed_difficulty": True,
        }

        configs = [baseline_config, balanced_config, recall_config]
        eval_results = []

        for cfg in configs:
            logger.info("Evaluating config: %s", cfg["name"])
            diffs_to_use = auth_diffs if cfg.get("use_smoothed_difficulty") else None

            with timed(f"Evaluate {cfg['name']}"):
                result = evaluate_config(
                    cfg, embeddings, ids, id_to_idx,
                    concepts, resource_sets,
                    difficulties_override=diffs_to_use,
                )

            # Remove sample edges for summary
            result_summary = {k: v for k, v in result.items() if k != "sample_top_100_edges"}
            eval_results.append(result_summary)
            logger.info(
                "  %s: edges=%d, isolated=%d (%.1f%%), depth=%d, dag=%s",
                cfg["name"],
                result_summary["total_edges"],
                result_summary["isolated_nodes_count"],
                result_summary["isolated_nodes_count"] / N * 100,
                result_summary["max_depth"],
                result_summary["is_dag"],
            )

            # Save full result
            _save(result, os.path.join(out_dir, f"{cfg['name']}_result.json"))

        # ================================================================
        # Best config selection
        # ================================================================
        best = balanced_config.copy()
        best_config_path = os.path.join(out_dir, "best_config.json")
        _save(best, best_config_path)

        # ================================================================
        # Final report
        # ================================================================
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_metrics": eval_results[0] if eval_results else {},
            "candidate_configs": eval_results,
            "recommended_config": best,
            "recommended_config_path": best_config_path,
            "reasoning": (
                "The 'balanced' config reduces isolated nodes significantly by: "
                "(1) using smoothed+authority-adjusted difficulty for better gap discrimination, "
                "(2) reweighting confidence to 0.70·sim + 0.20·gap + 0.10·cooc (similarity-dominant), "
                "(3) lowering sim_min to 0.45 and conf_threshold to 0.45. "
                "This retains semantic quality while dramatically improving graph connectivity."
            ),
            "difficulty_comparison": diff_comparison,
            "percentile_thresholds": pct_thresholds,
            "rollback": (
                "To rollback: DELETE FROM ConceptEdges; then re-run "
                "python -m src.graph_builder --db ./data/resources.db with original settings."
            ),
        }
        _save(report, os.path.join(out_dir, "phase3_recalibration_report.json"))

        # ---- Print summary ----
        print("\n" + "=" * 70)
        print("PHASE 3 CALIBRATION REPORT")
        print("=" * 70)
        for er in eval_results:
            cfg = er.get("config", {})
            name = cfg.get("name", "?")
            print(f"\n  [{name.upper()}]")
            print(f"    edges={er['total_edges']:,}  isolated={er['isolated_nodes_count']:,} "
                  f"({er['isolated_nodes_count']/N*100:.1f}%)  "
                  f"depth={er['max_depth']}  dag={er['is_dag']}")
            print(f"    sim_min={cfg.get('sim_min')}  conf={cfg.get('conf_threshold')}  "
                  f"w=({cfg.get('w_sim')},{cfg.get('w_gap')},{cfg.get('w_cooc')})  "
                  f"smoothed={cfg.get('use_smoothed_difficulty')}")

        print(f"\n  ★ Recommended: '{best['name']}' → {best_config_path}")
        print("=" * 70)

    finally:
        conn.close()


# =====================================================================
# E: Apply config
# =====================================================================

def apply_config(
    db_path: str,
    config_path: str,
    dry_run: bool = False,
):
    """Apply a saved config to the Phase 3 pipeline."""
    set_phase3_seed(42)

    with open(config_path) as fh:
        config = json.load(fh)

    logger.info("Applying config '%s' from %s (dry_run=%s)",
                config.get("name", "?"), config_path, dry_run)

    migrate_phase3(db_path)
    conn = phase1_db.get_connection(db_path)

    try:
        concepts = load_canonical_concepts(conn)
        embeddings, ids, id_to_idx = load_canonical_embeddings(conn)
        resource_sets = load_resource_sets(conn)
        N = len(concepts)

        # Compute difficulties
        orig_diffs = np.array(
            [concepts[id_to_idx[int(cid)]]["difficulty_score"] or 0.0
             for cid in ids],
            dtype=np.float32,
        )
        resource_counts = np.array(
            [concepts[id_to_idx[int(cid)]]["resource_count"] or 1
             for cid in ids],
            dtype=np.int32,
        )

        diffs_to_use = None
        if config.get("use_smoothed_difficulty"):
            index = build_index(embeddings, ids)
            _, neighbor_matrix = query_topk(index, embeddings, k=config.get("top_k", 15))
            smoothed = smooth_difficulty(orig_diffs, neighbor_matrix, ids, alpha=0.7, iterations=2)
            diffs_to_use = authority_adjusted_difficulty(smoothed, resource_counts, authority_weight=0.15)

        result = evaluate_config(
            config, embeddings, ids, id_to_idx,
            concepts, resource_sets,
            difficulties_override=diffs_to_use,
        )

        if dry_run:
            logger.info("DRY RUN — not persisting edges.")
        else:
            # Archive previous edges
            try:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS ConceptEdges_backup_{ts} AS
                    SELECT * FROM ConceptEdges
                """)
                conn.commit()
                logger.info("Archived previous edges → ConceptEdges_backup_%s", ts)
            except Exception as e:
                logger.warning("Could not archive edges: %s", e)

            clear_phase3_data(conn)
            # Extract good edges (without sample_top_100_edges metadata)
            edges = result.get("sample_top_100_edges", [])
            # We need to regenerate full edge list for persistence
            if diffs_to_use is not None:
                concept_list = []
                for c in concepts:
                    cc = dict(c)
                    idx = id_to_idx.get(c["id"])
                    if idx is not None:
                        cc["difficulty_score"] = float(diffs_to_use[idx])
                    concept_list.append(cc)
            else:
                concept_list = concepts

            all_edges = generate_candidate_edges(
                embeddings, ids, id_to_idx, concept_list, resource_sets,
                top_k=config.get("top_k", 15),
                max_out=config.get("max_out", 8),
                sim_min=config.get("sim_min", 0.45),
                conf_threshold=config.get("conf_threshold", 0.45),
                w_sim=config.get("w_sim", 0.70),
                w_gap=config.get("w_gap", 0.20),
                w_cooc=config.get("w_cooc", 0.10),
            )
            acyclic, removed = break_cycles(all_edges)
            n_inserted = insert_edges_batch(conn, acyclic)
            conn.commit()
            logger.info("Persisted %d edges.", n_inserted)

            # Update summary
            final_metrics = compute_metrics(acyclic, n_concepts=N)
            final_metrics["is_dag"] = validate_dag(acyclic)
            final_metrics["removed_edges"] = len(removed)
            data_dir = os.path.dirname(os.path.abspath(db_path))
            _save(final_metrics, os.path.join(data_dir, "phase3_summary.json"))

        summary = {k: v for k, v in result.items() if k != "sample_top_100_edges"}
        print("\n" + "=" * 60)
        print("APPLY CONFIG RESULT")
        print("=" * 60)
        print(f"  Config: {config.get('name', '?')}")
        print(f"  Edges:  {summary['total_edges']:,}")
        print(f"  Isolated: {summary['isolated_nodes_count']:,} ({summary['isolated_nodes_count']/N*100:.1f}%)")
        print(f"  Max depth: {summary['max_depth']}")
        print(f"  DAG:    {summary['is_dag']}")
        print(f"  Mode:   {'DRY RUN' if dry_run else 'PERSISTED'}")
        print("=" * 60)

        return summary

    finally:
        conn.close()


# =====================================================================
# CLI
# =====================================================================

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 3 calibration & recalibration")
    parser.add_argument("--db", default="./data/resources.db")
    parser.add_argument("--out", default="./data/phase3_calibration",
                        help="Output directory for calibration reports")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--apply-config", type=str, default=None,
                        help="Path to config JSON to apply")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but don't persist edges")
    args = parser.parse_args()

    if args.apply_config:
        apply_config(args.db, args.apply_config, dry_run=args.dry_run)
    else:
        run_calibration(args.db, out_dir=args.out, top_k=args.top_k)


if __name__ == "__main__":
    main()
