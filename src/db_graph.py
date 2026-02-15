"""
Phase 3 database helpers: ConceptEdges table + canonical embedding loader.

Provides:
- ``migrate_phase3``  — create the ``ConceptEdges`` table with indexes.
- ``load_canonical_embeddings`` — aggregate ExtractedConcepts embeddings
  per canonical_id via mean-pooling.
- CRUD helpers for edges.
"""

import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.utils import blob_to_embedding

logger = logging.getLogger(__name__)

# =========================================================================
# Schema
# =========================================================================

_CREATE_CONCEPT_EDGES = """\
CREATE TABLE IF NOT EXISTS ConceptEdges (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_concept_id  INTEGER NOT NULL,
    target_concept_id  INTEGER NOT NULL,
    similarity         REAL    NOT NULL,
    confidence         REAL    NOT NULL,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_concept_id) REFERENCES CanonicalConcepts(id),
    FOREIGN KEY (target_concept_id) REFERENCES CanonicalConcepts(id),
    UNIQUE(source_concept_id, target_concept_id)
);
"""

_CREATE_IDX_SOURCE = """\
CREATE INDEX IF NOT EXISTS idx_edges_source
    ON ConceptEdges(source_concept_id);
"""

_CREATE_IDX_TARGET = """\
CREATE INDEX IF NOT EXISTS idx_edges_target
    ON ConceptEdges(target_concept_id);
"""


# =========================================================================
# Migration
# =========================================================================


def migrate_phase3(db_path: str) -> None:
    """Create (or verify) the ``ConceptEdges`` table + indexes."""
    from src.db import get_connection

    conn = get_connection(db_path)
    try:
        conn.execute(_CREATE_CONCEPT_EDGES)
        conn.execute(_CREATE_IDX_SOURCE)
        conn.execute(_CREATE_IDX_TARGET)
        conn.commit()
        logger.info("Phase 3 migration OK.")
    finally:
        conn.close()


# =========================================================================
# Load canonical concept data
# =========================================================================


def load_canonical_concepts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Load all CanonicalConcepts rows."""
    rows = conn.execute(
        """SELECT id, canonical_concept, difficulty_score, resource_count
           FROM CanonicalConcepts ORDER BY id"""
    ).fetchall()
    return [dict(r) for r in rows]


def load_canonical_embeddings(
    conn: sqlite3.Connection,
    dim: int = 384,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Compute canonical-concept embeddings via mean-pooling.

    Groups ``ExtractedConcepts.embedding`` by ``canonical_id`` and
    averages (mean-pools) each group. Only canonical IDs that have
    at least one non-null embedding are included.

    Returns:
        Tuple of:
        - ``embeddings``: ``(N, dim)`` float32, L2-normalised
        - ``ids``: ``(N,)`` int64 array of canonical concept IDs
        - ``id_to_idx``: mapping from canonical ID → row index in the arrays
    """
    rows = conn.execute(
        """SELECT canonical_id, embedding
           FROM ExtractedConcepts
           WHERE canonical_id IS NOT NULL AND embedding IS NOT NULL"""
    ).fetchall()

    groups: Dict[int, List[np.ndarray]] = defaultdict(list)
    for row in rows:
        cid = row[0]
        emb = blob_to_embedding(row[1], dim=dim)
        groups[cid].append(emb)

    sorted_ids = sorted(groups.keys())
    embeddings = np.zeros((len(sorted_ids), dim), dtype=np.float32)
    for idx, cid in enumerate(sorted_ids):
        stacked = np.stack(groups[cid])
        embeddings[idx] = stacked.mean(axis=0)

    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms

    ids = np.array(sorted_ids, dtype=np.int64)
    id_to_idx = {cid: idx for idx, cid in enumerate(sorted_ids)}

    logger.info(
        "Loaded %d canonical embeddings (dim=%d) via mean-pooling.",
        len(ids), dim,
    )
    return embeddings, ids, id_to_idx


def load_resource_sets(conn: sqlite3.Connection) -> Dict[int, Set[int]]:
    """Return ``{canonical_id: set(resource_ids)}`` for co-occurrence scoring."""
    rows = conn.execute(
        """SELECT canonical_id, resource_id
           FROM ExtractedConcepts
           WHERE canonical_id IS NOT NULL"""
    ).fetchall()
    result: Dict[int, Set[int]] = defaultdict(set)
    for row in rows:
        result[row[0]].add(row[1])
    return result


# =========================================================================
# Edge persistence
# =========================================================================


def clear_phase3_data(conn: sqlite3.Connection) -> None:
    """Wipe ConceptEdges for idempotent re-runs."""
    conn.execute("DELETE FROM ConceptEdges;")
    conn.commit()
    logger.info("ConceptEdges cleared for idempotent re-run.")


def insert_edges_batch(
    conn: sqlite3.Connection,
    edges: List[Dict[str, Any]],
) -> int:
    """Insert edges in a single transaction. Skips self-loops.

    Each dict: ``{source_id, target_id, similarity, confidence}``.
    Returns number of rows inserted.
    """
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for e in edges:
        if e["source_id"] == e["target_id"]:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO ConceptEdges
                   (source_concept_id, target_concept_id, similarity,
                    confidence, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (e["source_id"], e["target_id"], e["similarity"],
             e["confidence"], now),
        )
        count += 1
    conn.commit()
    return count


def get_all_edges(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all ConceptEdges rows."""
    rows = conn.execute(
        "SELECT * FROM ConceptEdges ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]
