"""
Phase 2 CLI: Concept Extraction + Canonicalization.

Usage::

    python -m src.extract_concepts \\
        --db ./data/resources.db \\
        --batch-size 64 \\
        --cluster-threshold 0.85

Reads ``RawResources`` (status='ok'), extracts candidate concepts,
embeds them, clusters into canonical concepts, scores difficulty,
and writes results to the database + ``phase2_summary.json``.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from src import db
from src.canonicalization.clusterer import canonicalize
from src.extractors.spacy_extractor import extract_all
from src.models import Phase2Summary
from src.utils import embedding_to_blob, set_global_seed, setup_logging

logger = logging.getLogger(__name__)


# =========================================================================
# Persistence
# =========================================================================


def _persist_results(
    conn,
    extraction_results,
    clusters,
    index,
    embeddings,
) -> Dict[int, int]:
    """Write Phase 2 results to the database in a single transaction.

    Returns:
        Mapping ``{extracted_concept_row_id: canonical_concept_row_id}``.
    """
    # --- Insert canonical concepts ---
    canonical_id_map: Dict[str, int] = {}  # canonical_label → db id
    for cl in clusters:
        cc_id = db.insert_canonical_concept(
            conn,
            canonical_concept=cl.canonical_label,
            difficulty_score=cl.difficulty_score,
            difficulty_bucket=cl.difficulty_bucket,
            example_sentence=cl.example_sentence,
            resource_count=len(cl.resource_ids),
        )
        canonical_id_map[cl.canonical_label] = cc_id

    # --- Build label → canonical_label lookup ---
    text_to_canonical: Dict[str, str] = {}
    for cl in clusters:
        for member in cl.members:
            text_to_canonical[member] = cl.canonical_label

    # --- Insert extracted concepts ---
    ec_rows: List[Dict] = []
    for cr in extraction_results:
        for concept, sentence in cr.candidates:
            idx = index.text_to_idx.get(concept)
            emb_blob = (
                embedding_to_blob(embeddings[idx]) if idx is not None else None
            )
            ec_rows.append({
                "resource_id": cr.resource_id,
                "concept": concept,
                "sentence": sentence,
                "embedding": emb_blob,
            })

    ec_ids = db.insert_extracted_concepts_batch(conn, ec_rows)

    # --- Update canonical_id on extracted concepts ---
    ec_canonical_map: Dict[int, int] = {}
    for i, ec_id in enumerate(ec_ids):
        concept_text = ec_rows[i]["concept"]
        canon_label = text_to_canonical.get(concept_text)
        if canon_label and canon_label in canonical_id_map:
            ec_canonical_map[ec_id] = canonical_id_map[canon_label]

    if ec_canonical_map:
        db.update_canonical_ids(conn, ec_canonical_map)

    logger.info(
        "Persisted %d extracted concepts, %d canonical concepts.",
        len(ec_ids), len(canonical_id_map),
    )
    return ec_canonical_map


# =========================================================================
# Summary
# =========================================================================


def _build_summary(
    n_resources: int,
    n_candidates: int,
    clusters,
) -> Phase2Summary:
    """Build the Phase2Summary from pipeline results."""
    dist = {"beginner": 0, "intermediate": 0, "advanced": 0}
    for cl in clusters:
        bucket = cl.difficulty_bucket
        if bucket in dist:
            dist[bucket] += 1

    avg = n_candidates / n_resources if n_resources > 0 else 0.0

    return Phase2Summary(
        resources_processed=n_resources,
        candidate_concepts=n_candidates,
        canonical_concepts=len(clusters),
        avg_concepts_per_resource=round(avg, 2),
        difficulty_distribution=dist,
    )


# =========================================================================
# Pipeline orchestration
# =========================================================================


def run_pipeline(
    db_path: str = "./data/resources.db",
    batch_size: int = 64,
    cluster_threshold: float = 0.85,
) -> Phase2Summary:
    """Execute the full Phase 2 pipeline.

    Args:
        db_path: Path to the SQLite database.
        batch_size: Embedding batch size.
        cluster_threshold: Cosine similarity threshold for clustering.

    Returns:
        ``Phase2Summary`` with statistics.
    """
    set_global_seed(42)

    # --- Migrate ---
    db.migrate_phase2(db_path)
    conn = db.get_connection(db_path)

    try:
        # --- Idempotent re-run ---
        db.clear_phase2_data(conn)

        # --- Load resources ---
        resources = db.get_ok_resources(conn)
        if not resources:
            logger.error(
                "No resources with status='ok' found in %s. "
                "Run Phase 1 ingestion first (python -m src.ingest …).",
                db_path,
            )
            raise SystemExit(1)

        logger.info("Loaded %d resources with status='ok'.", len(resources))

        # --- 1. Extract candidates ---
        t0 = time.monotonic()
        extraction_results = extract_all(resources)
        n_candidates = sum(len(cr.candidates) for cr in extraction_results)
        elapsed = time.monotonic() - t0
        logger.info(
            "Extraction: %d candidates from %d resources in %.1fs.",
            n_candidates, len(resources), elapsed,
        )

        if n_candidates == 0:
            logger.warning("No candidates extracted — nothing to cluster.")
            return _build_summary(len(resources), 0, [])

        # --- 2+3+4. Canonicalize (embed → cluster → score) ---
        t0 = time.monotonic()
        clusters, index, embeddings = canonicalize(
            extraction_results,
            cluster_threshold=cluster_threshold,
            batch_size=batch_size,
        )
        elapsed = time.monotonic() - t0
        mem_mb = embeddings.nbytes / (1024 * 1024) if embeddings.size else 0
        logger.info(
            "Canonicalization: %d clusters in %.1fs "
            "(embeddings: %.1f MB).",
            len(clusters), elapsed, mem_mb,
        )

        # --- 5. Persist ---
        t0 = time.monotonic()
        _persist_results(conn, extraction_results, clusters, index, embeddings)
        elapsed = time.monotonic() - t0
        logger.info("Persistence: %.1fs.", elapsed)

        # --- 6. Summary ---
        summary = _build_summary(len(resources), n_candidates, clusters)

        # Write summary JSON
        data_dir = os.path.dirname(os.path.abspath(db_path))
        summary_path = os.path.join(data_dir, "phase2_summary.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(summary.model_dump_json(indent=2))
        logger.info("📄 Summary written to %s", summary_path)

        return summary

    finally:
        conn.close()


# =========================================================================
# CLI
# =========================================================================


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.extract_concepts",
        description="Phase 2: Concept Extraction + Canonicalization.",
    )
    parser.add_argument(
        "--db",
        default="./data/resources.db",
        help="Path to the SQLite database (default: ./data/resources.db).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for merging (default: 0.85).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry-point."""
    setup_logging()
    args = _parse_args(argv)

    logger.info(
        "Phase 2 starting — db=%s, batch_size=%d, cluster_threshold=%.2f",
        args.db, args.batch_size, args.cluster_threshold,
    )

    summary = run_pipeline(
        db_path=args.db,
        batch_size=args.batch_size,
        cluster_threshold=args.cluster_threshold,
    )

    logger.info(
        "✅ Phase 2 complete — resources=%d, candidates=%d, "
        "canonical=%d, difficulty=%s",
        summary.resources_processed,
        summary.candidate_concepts,
        summary.canonical_concepts,
        summary.difficulty_distribution,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
