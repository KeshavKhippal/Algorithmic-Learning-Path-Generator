"""
Phase 3 CLI: Concept Graph Construction (DAG).

Usage::

    python -m src.graph_builder \\
        --db ./data/resources.db \\
        --top-k 15 --max-out 8 \\
        --sim-min 0.50 --conf-threshold 0.55

Reads ``CanonicalConcepts`` + ``ExtractedConcepts`` embeddings,
builds a FAISS index, generates directed prerequisite edges,
enforces acyclicity, and persists the DAG to ``ConceptEdges``.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set

import numpy as np

from src import db as phase1_db
from src.dag_validator import break_cycles, compute_metrics, validate_dag
from src.db_graph import (
    clear_phase3_data,
    insert_edges_batch,
    load_canonical_concepts,
    load_canonical_embeddings,
    load_resource_sets,
    migrate_phase3,
)
from src.faiss_index import build_index, query_topk
from src.utils import setup_logging
from src.utils_graph import set_phase3_seed, timed

logger = logging.getLogger(__name__)


# =========================================================================
# Edge generation
# =========================================================================


def generate_candidate_edges(
    embeddings: np.ndarray,
    ids: np.ndarray,
    id_to_idx: Dict[int, int],
    concepts: List[Dict[str, Any]],
    resource_sets: Dict[int, Set[int]],
    top_k: int = 15,
    max_out: int = 8,
    sim_min: float = 0.50,
    conf_threshold: float = 0.55,
    use_gpu: bool = False,
    w_sim: float = 0.60,
    w_gap: float = 0.30,
    w_cooc: float = 0.10,
) -> List[Dict[str, Any]]:
    """Generate scored, direction-enforced candidate edges.

    Args:
        embeddings: ``(N, D)`` normalised canonical embeddings.
        ids: ``(N,)`` canonical concept IDs.
        id_to_idx: Mapping canonical_id → row index.
        concepts: List of canonical concept dicts with ``id``,
                  ``difficulty_score``, ``resource_count``.
        resource_sets: ``{canonical_id: set(resource_ids)}``.
        top_k: FAISS neighbors per concept.
        max_out: Max outgoing edges per source.
        sim_min: Minimum cosine similarity for an edge.
        conf_threshold: Minimum confidence for an edge.
        use_gpu: Use GPU FAISS if available.
        w_sim: Weight for similarity in confidence formula.
        w_gap: Weight for difficulty gap in confidence formula.
        w_cooc: Weight for co-occurrence bonus in confidence formula.

    Returns:
        List of edge dicts: ``{source_id, target_id, similarity, confidence}``.
    """
    # Build concept lookup
    concept_map: Dict[int, Dict[str, Any]] = {c["id"]: c for c in concepts}

    # Build FAISS index and query
    with timed("FAISS index build"):
        index = build_index(embeddings, ids, use_gpu=use_gpu)

    with timed("FAISS top-k query"):
        similarities, neighbor_ids = query_topk(index, embeddings, k=top_k)

    # Generate edges
    all_candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row_idx in range(len(ids)):
        src_id = int(ids[row_idx])
        src = concept_map.get(src_id)
        if src is None:
            continue
        src_diff = src["difficulty_score"] or 0.0

        for col in range(similarities.shape[1]):
            tgt_id = int(neighbor_ids[row_idx, col])
            sim = float(similarities[row_idx, col])

            # Skip self, padding, and below-threshold
            if tgt_id == src_id or tgt_id < 0:
                continue
            if sim < sim_min:
                continue

            tgt = concept_map.get(tgt_id)
            if tgt is None:
                continue
            tgt_diff = tgt["difficulty_score"] or 0.0

            # Direction: source must be easier than target
            if src_diff >= tgt_diff:
                continue

            # Co-occurrence bonus
            src_res = resource_sets.get(src_id, set())
            tgt_res = resource_sets.get(tgt_id, set())
            max_rc = max(len(src_res), len(tgt_res), 1)
            cooc = len(src_res & tgt_res) / max_rc

            # Difficulty gap
            diff_gap = min((tgt_diff - src_diff) / 1.0, 1.0)

            # Confidence (configurable weights)
            confidence = w_sim * sim + w_gap * diff_gap + w_cooc * cooc
            if confidence < conf_threshold:
                continue

            all_candidates[src_id].append({
                "source_id": src_id,
                "target_id": tgt_id,
                "similarity": round(sim, 6),
                "confidence": round(confidence, 6),
            })

    # Keep top max_out per source by confidence
    final_edges: List[Dict[str, Any]] = []
    for src_id, edges in all_candidates.items():
        edges.sort(key=lambda e: e["confidence"], reverse=True)
        final_edges.extend(edges[:max_out])

    logger.info(
        "Generated %d candidate edges from %d sources.",
        len(final_edges), len(all_candidates),
    )
    return final_edges


# =========================================================================
# Pipeline
# =========================================================================


def run_pipeline(
    db_path: str = "./data/resources.db",
    top_k: int = 15,
    max_out: int = 8,
    sim_min: float = 0.50,
    conf_threshold: float = 0.55,
    use_gpu: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute the full Phase 3 pipeline.

    Returns:
        Summary dict (same schema as ``phase3_summary.json``).
    """
    set_phase3_seed(42)

    # --- Migrate ---
    migrate_phase3(db_path)
    conn = phase1_db.get_connection(db_path)

    try:
        # --- Load concepts ---
        with timed("Load canonical concepts"):
            concepts = load_canonical_concepts(conn)

        if not concepts:
            logger.error(
                "No CanonicalConcepts found in %s. Run Phase 2 first.", db_path
            )
            raise SystemExit(1)
        logger.info("Loaded %d canonical concepts.", len(concepts))

        # --- Load embeddings ---
        with timed("Load canonical embeddings"):
            embeddings, ids, id_to_idx = load_canonical_embeddings(conn)

        if len(ids) == 0:
            logger.error("No canonical embeddings found. Run Phase 2 first.")
            raise SystemExit(1)

        logger.info("Embedding matrix: N=%d, D=%d.", *embeddings.shape)

        # --- Load resource sets ---
        resource_sets = load_resource_sets(conn)

        # --- Generate edges ---
        with timed("Edge generation"):
            candidate_edges = generate_candidate_edges(
                embeddings, ids, id_to_idx, concepts, resource_sets,
                top_k=top_k, max_out=max_out, sim_min=sim_min,
                conf_threshold=conf_threshold, use_gpu=use_gpu,
            )

        # --- Cycle-breaking ---
        with timed("Cycle-breaking"):
            acyclic_edges, removed_edges = break_cycles(candidate_edges)

        # --- Save removed edges ---
        data_dir = os.path.dirname(os.path.abspath(db_path))
        removed_path = os.path.join(data_dir, "phase3_removed_edges.json")
        with open(removed_path, "w", encoding="utf-8") as fh:
            json.dump(removed_edges, fh, indent=2, default=str)
        logger.info(
            "Removed %d edge(s) for acyclicity → %s",
            len(removed_edges), removed_path,
        )

        # --- Persist ---
        if not dry_run:
            with timed("Persistence"):
                clear_phase3_data(conn)
                n_inserted = insert_edges_batch(conn, acyclic_edges)
            logger.info("Persisted %d edges.", n_inserted)
        else:
            logger.info("Dry run — edges NOT persisted.")

        # --- Validate ---
        is_dag = validate_dag(acyclic_edges)
        if not is_dag:
            logger.error("❌ Graph is NOT acyclic after cycle-breaking!")
        else:
            logger.info("✅ Graph is acyclic (topological sort OK).")

        # --- Metrics ---
        metrics = compute_metrics(acyclic_edges, len(concepts))
        metrics["is_dag"] = is_dag
        metrics["removed_edges"] = len(removed_edges)

        # --- Summary ---
        summary_path = os.path.join(data_dir, "phase3_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("📄 Summary → %s", summary_path)

        return metrics

    finally:
        conn.close()


# =========================================================================
# CLI
# =========================================================================


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.graph_builder",
        description="Phase 3: Concept Graph Construction (DAG).",
    )
    parser.add_argument("--db", default="./data/resources.db")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-out", type=int, default=8)
    parser.add_argument("--sim-min", type=float, default=0.50)
    parser.add_argument("--conf-threshold", type=float, default=0.55)
    parser.add_argument("--use-gpu-faiss", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Calibration flags
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="Run calibration sweep (no edge persistence).",
    )
    parser.add_argument(
        "--apply-config", type=str, default=None,
        help="Apply a saved calibration config JSON.",
    )
    parser.add_argument(
        "--save-config", type=str, default=None,
        help="Save current settings to a config JSON.",
    )
    parser.add_argument(
        "--out", type=str, default="./data/phase3_calibration",
        help="Output directory for calibration reports.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry-point."""
    setup_logging()
    args = _parse_args(argv)

    # --save-config: just dump settings and exit
    if args.save_config:
        config = {
            "top_k": args.top_k, "max_out": args.max_out,
            "sim_min": args.sim_min, "conf_threshold": args.conf_threshold,
            "w_sim": 0.60, "w_gap": 0.30, "w_cooc": 0.10,
            "use_smoothed_difficulty": False,
        }
        os.makedirs(os.path.dirname(args.save_config) or ".", exist_ok=True)
        with open(args.save_config, "w") as fh:
            json.dump(config, fh, indent=2)
        logger.info("Config saved → %s", args.save_config)
        return

    # --calibrate-only: run calibration sweep
    if args.calibrate_only:
        from src.phase3_calibrate import run_calibration
        run_calibration(db_path=args.db, out_dir=args.out, top_k=args.top_k)
        return

    # --apply-config: apply a calibration config
    if args.apply_config:
        from src.phase3_calibrate import apply_config
        apply_config(args.db, args.apply_config, dry_run=args.dry_run)
        return

    # Standard Phase 3 pipeline
    logger.info(
        "Phase 3 starting — db=%s, top_k=%d, max_out=%d, "
        "sim_min=%.2f, conf=%.2f, gpu=%s, dry_run=%s",
        args.db, args.top_k, args.max_out,
        args.sim_min, args.conf_threshold,
        args.use_gpu_faiss, args.dry_run,
    )

    metrics = run_pipeline(
        db_path=args.db,
        top_k=args.top_k,
        max_out=args.max_out,
        sim_min=args.sim_min,
        conf_threshold=args.conf_threshold,
        use_gpu=args.use_gpu_faiss,
        dry_run=args.dry_run,
    )

    logger.info(
        "✅ Phase 3 complete — edges=%d, avg_out=%.2f, "
        "max_depth=%d, isolated=%d, removed=%d",
        metrics["total_edges"],
        metrics["avg_out_degree"],
        metrics["max_depth"],
        metrics["isolated_nodes_count"],
        metrics["removed_edges"],
    )
    sys.exit(0)


if __name__ == "__main__":
    main()

