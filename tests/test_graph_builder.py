"""
pytest suite for Phase 3: Concept Graph Construction (DAG).

All tests use deterministic seeded data — no network or GPU needed.
"""

import json
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db as phase1_db
from src.db_graph import (
    clear_phase3_data,
    get_all_edges,
    insert_edges_batch,
    load_canonical_concepts,
    migrate_phase3,
)
from src.dag_validator import break_cycles, compute_metrics, validate_dag
from src.faiss_index import build_index, query_topk
from src.graph_builder import generate_candidate_edges, run_pipeline
from src.utils_graph import set_phase3_seed


# =========================================================================
# Helpers
# =========================================================================


def _make_embeddings(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Create deterministic normalised embeddings."""
    np.random.seed(seed)
    emb = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def _make_similar_pair(dim: int = 384, seed: int = 42):
    """Create two vectors with high cosine similarity (~0.99)."""
    np.random.seed(seed)
    base = np.random.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    noise = np.random.randn(dim).astype(np.float32) * 0.01
    other = base + noise
    other /= np.linalg.norm(other)
    return base, other


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def tmp_db(tmp_path):
    """Return a DB path inside a temporary directory."""
    return str(tmp_path / "test_phase3.db")


@pytest.fixture()
def seeded_db(tmp_db):
    """DB pre-seeded with Phase 1 + Phase 2 tables and 8 sample concepts."""
    from src.utils import embedding_to_blob

    # Migrate all phases
    phase1_db.migrate_db(tmp_db)
    from src.db import migrate_phase2
    migrate_phase2(tmp_db)
    migrate_phase3(tmp_db)

    conn = phase1_db.get_connection(tmp_db)

    # Insert sample resources
    for i in range(1, 4):
        phase1_db.insert_resource(
            conn,
            url=f"https://example.com/res-{i}",
            content_type="article",
            title=f"Resource {i}",
            raw_text=f"Sample text for resource {i}.",
            status="ok",
        )

    # Load sample Phase 3 data
    data_path = os.path.join(
        os.path.dirname(__file__), "sample_phase3_data.json"
    )
    with open(data_path) as fh:
        concepts = json.load(fh)

    # Insert canonical concepts
    from src.db import insert_canonical_concept
    for c in concepts:
        insert_canonical_concept(
            conn,
            canonical_concept=c["canonical_concept"],
            difficulty_score=c["difficulty_score"],
            difficulty_bucket=(
                "beginner" if c["difficulty_score"] < 0.33
                else "intermediate" if c["difficulty_score"] < 0.66
                else "advanced"
            ),
            example_sentence=f"Example for {c['canonical_concept']}.",
            resource_count=c["resource_count"],
        )

    # Insert extracted concepts with embeddings (linking to canonical)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    for c in concepts:
        np.random.seed(c["embedding_seed"])
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        blob = embedding_to_blob(emb)

        # Assign to resources round-robin based on resource_count
        for r in range(1, min(c["resource_count"], 3) + 1):
            conn.execute(
                """INSERT INTO ExtractedConcepts
                       (resource_id, concept, canonical_id, sentence,
                        embedding, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (r, c["canonical_concept"], c["id"],
                 f"Sentence about {c['canonical_concept']}.",
                 blob, now),
            )
    conn.commit()
    conn.close()
    return tmp_db


# =========================================================================
# Test: FAISS Index
# =========================================================================


class TestFaissIndex:
    """Tests for FAISS index construction and query."""

    def test_build_and_query_deterministic(self):
        """Build index, query top-k, get deterministic results."""
        set_phase3_seed(42)
        emb = _make_embeddings(10, dim=384)
        ids = np.arange(100, 110, dtype=np.int64)

        index = build_index(emb, ids)
        assert index.ntotal == 10

        sims, nids = query_topk(index, emb[:1], k=3)
        assert sims.shape == (1, 3)
        assert nids.shape == (1, 3)
        # First result should be self (highest similarity)
        assert nids[0, 0] == 100
        assert abs(sims[0, 0] - 1.0) < 0.01

    def test_similar_vectors_found(self):
        """Two highly similar vectors should be each other's top neighbor."""
        base, other = _make_similar_pair()
        emb = np.stack([base, other])
        ids = np.array([1, 2], dtype=np.int64)

        index = build_index(emb, ids)
        sims, nids = query_topk(index, emb, k=2)

        # ID=1's top neighbor (after self) should be ID=2
        assert nids[0, 1] == 2
        assert sims[0, 1] > 0.95

    def test_empty_index(self):
        """Empty embedding array should work without crash."""
        emb = np.empty((0, 384), dtype=np.float32)
        ids = np.empty(0, dtype=np.int64)
        index = build_index(emb, ids)
        assert index.ntotal == 0


# =========================================================================
# Test: Edge Generation Rules
# =========================================================================


class TestEdgeGenerationRules:
    """Tests for difficulty ordering and similarity thresholds."""

    def test_difficulty_direction_enforced(self):
        """Edges must flow from lower to higher difficulty only."""
        set_phase3_seed(42)
        # Make two similar vectors
        base, other = _make_similar_pair()
        emb = np.stack([base, other])
        ids = np.array([1, 2], dtype=np.int64)
        id_to_idx = {1: 0, 2: 1}

        # Concept 1 harder than concept 2 → no edge 1→2,
        # but edge 2→1 allowed
        concepts = [
            {"id": 1, "difficulty_score": 0.8, "resource_count": 5},
            {"id": 2, "difficulty_score": 0.2, "resource_count": 5},
        ]
        resource_sets = {1: {1, 2}, 2: {1, 2, 3}}

        edges = generate_candidate_edges(
            emb, ids, id_to_idx, concepts, resource_sets,
            top_k=2, sim_min=0.0, conf_threshold=0.0,
        )
        for e in edges:
            src_diff = next(
                c["difficulty_score"] for c in concepts if c["id"] == e["source_id"]
            )
            tgt_diff = next(
                c["difficulty_score"] for c in concepts if c["id"] == e["target_id"]
            )
            assert src_diff < tgt_diff, (
                f"Edge {e['source_id']}→{e['target_id']}: "
                f"src_diff={src_diff} >= tgt_diff={tgt_diff}"
            )

    def test_similarity_minimum_enforced(self):
        """No edge with similarity below sim_min."""
        set_phase3_seed(42)
        emb = _make_embeddings(5)
        ids = np.arange(1, 6, dtype=np.int64)
        id_to_idx = {i: i - 1 for i in range(1, 6)}
        concepts = [
            {"id": i, "difficulty_score": i * 0.15, "resource_count": 3}
            for i in range(1, 6)
        ]
        resource_sets = {i: {1} for i in range(1, 6)}

        edges = generate_candidate_edges(
            emb, ids, id_to_idx, concepts, resource_sets,
            top_k=5, sim_min=0.90, conf_threshold=0.0,
        )
        for e in edges:
            assert e["similarity"] >= 0.90

    def test_no_self_loops(self):
        """No edge where source == target."""
        set_phase3_seed(42)
        emb = _make_embeddings(3)
        ids = np.arange(1, 4, dtype=np.int64)
        id_to_idx = {1: 0, 2: 1, 3: 2}
        concepts = [
            {"id": 1, "difficulty_score": 0.1, "resource_count": 5},
            {"id": 2, "difficulty_score": 0.5, "resource_count": 3},
            {"id": 3, "difficulty_score": 0.9, "resource_count": 1},
        ]
        resource_sets = {1: {1}, 2: {1}, 3: {1}}

        edges = generate_candidate_edges(
            emb, ids, id_to_idx, concepts, resource_sets,
            top_k=3, sim_min=0.0, conf_threshold=0.0,
        )
        for e in edges:
            assert e["source_id"] != e["target_id"]


# =========================================================================
# Test: Cycle Detection & Breaking
# =========================================================================


class TestCycleDetectionAndBreaking:
    """Tests for the greedy cycle-breaking algorithm."""

    def test_acyclic_input_unchanged(self):
        """No cycles → no edges removed."""
        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.8},
            {"source_id": 2, "target_id": 3, "similarity": 0.8, "confidence": 0.7},
        ]
        acyclic, removed = break_cycles(edges)
        assert len(removed) == 0
        assert len(acyclic) == 2

    def test_cycle_broken_by_min_confidence(self):
        """Cycle should be broken by removing the lowest-confidence edge."""
        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.8},
            {"source_id": 2, "target_id": 3, "similarity": 0.8, "confidence": 0.7},
            {"source_id": 3, "target_id": 1, "similarity": 0.7, "confidence": 0.3},  # weakest
        ]
        acyclic, removed = break_cycles(edges)
        assert len(removed) == 1
        assert removed[0]["source_id"] == 3
        assert removed[0]["target_id"] == 1
        assert removed[0]["reason"] == "cycle_break"
        assert validate_dag(acyclic)

    def test_multiple_cycles(self):
        """Multiple overlapping cycles should all be resolved."""
        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.9},
            {"source_id": 2, "target_id": 3, "similarity": 0.8, "confidence": 0.8},
            {"source_id": 3, "target_id": 1, "similarity": 0.7, "confidence": 0.3},
            {"source_id": 2, "target_id": 4, "similarity": 0.8, "confidence": 0.7},
            {"source_id": 4, "target_id": 2, "similarity": 0.6, "confidence": 0.2},
        ]
        acyclic, removed = break_cycles(edges)
        assert validate_dag(acyclic)
        assert len(removed) >= 2

    def test_validate_dag_true(self):
        """A simple chain is a DAG."""
        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.8},
            {"source_id": 2, "target_id": 3, "similarity": 0.8, "confidence": 0.7},
        ]
        assert validate_dag(edges) is True

    def test_validate_dag_false(self):
        """A cycle is NOT a DAG."""
        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.8},
            {"source_id": 2, "target_id": 1, "similarity": 0.8, "confidence": 0.7},
        ]
        assert validate_dag(edges) is False


# =========================================================================
# Test: Metrics
# =========================================================================


class TestMetrics:
    """Tests for graph metric computation."""

    def test_metrics_basic(self):
        """Metrics for a simple chain graph."""
        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.8},
            {"source_id": 2, "target_id": 3, "similarity": 0.8, "confidence": 0.7},
        ]
        m = compute_metrics(edges, n_concepts=5)
        assert m["total_edges"] == 2
        assert m["max_depth"] == 2
        assert m["isolated_nodes_count"] == 2  # 5 total - 3 in graph

    def test_metrics_empty(self):
        """Zero edges → all isolated."""
        m = compute_metrics([], n_concepts=10)
        assert m["total_edges"] == 0
        assert m["max_depth"] == 0
        assert m["isolated_nodes_count"] == 10


# =========================================================================
# Test: DB Persistence
# =========================================================================


class TestPersistence:
    """Tests for edge persistence + idempotency."""

    def test_table_created(self, tmp_db):
        """Phase 3 migration creates ConceptEdges table."""
        phase1_db.migrate_db(tmp_db)
        migrate_phase3(tmp_db)
        conn = phase1_db.get_connection(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert "ConceptEdges" in {r[0] for r in tables}

    def test_insert_and_retrieve(self, tmp_db):
        """Insert edges and retrieve them."""
        phase1_db.migrate_db(tmp_db)
        migrate_phase3(tmp_db)
        conn = phase1_db.get_connection(tmp_db)

        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.8},
            {"source_id": 2, "target_id": 3, "similarity": 0.8, "confidence": 0.7},
        ]
        n = insert_edges_batch(conn, edges)
        assert n == 2

        rows = get_all_edges(conn)
        conn.close()
        assert len(rows) == 2

    def test_idempotent_clear_and_reinsert(self, tmp_db):
        """Clear + reinsert yields same count."""
        phase1_db.migrate_db(tmp_db)
        migrate_phase3(tmp_db)
        conn = phase1_db.get_connection(tmp_db)

        edges = [
            {"source_id": 1, "target_id": 2, "similarity": 0.9, "confidence": 0.8},
        ]
        insert_edges_batch(conn, edges)
        assert len(get_all_edges(conn)) == 1

        clear_phase3_data(conn)
        insert_edges_batch(conn, edges)
        assert len(get_all_edges(conn)) == 1
        conn.close()

    def test_self_loop_rejected(self, tmp_db):
        """Self-loops are skipped by insert_edges_batch."""
        phase1_db.migrate_db(tmp_db)
        migrate_phase3(tmp_db)
        conn = phase1_db.get_connection(tmp_db)

        edges = [
            {"source_id": 1, "target_id": 1, "similarity": 1.0, "confidence": 1.0},
        ]
        n = insert_edges_batch(conn, edges)
        assert n == 0
        assert len(get_all_edges(conn)) == 0
        conn.close()


# =========================================================================
# Test: Integration (full pipeline on seeded DB)
# =========================================================================


class TestIntegration:
    """Integration test using seeded DB with 8 sample concepts."""

    def test_full_pipeline(self, seeded_db):
        """Run the full pipeline on sample data and verify output."""
        metrics = run_pipeline(
            db_path=seeded_db,
            top_k=5,
            max_out=4,
            sim_min=0.0,
            conf_threshold=0.0,
        )

        assert metrics["is_dag"] is True
        assert metrics["total_concepts"] == 8
        assert metrics["total_edges"] >= 0

        # Verify DB has edges
        conn = phase1_db.get_connection(seeded_db)
        rows = get_all_edges(conn)
        conn.close()
        assert len(rows) == metrics["total_edges"]

        # Verify summary file
        data_dir = os.path.dirname(seeded_db)
        summary_path = os.path.join(data_dir, "phase3_summary.json")
        assert os.path.isfile(summary_path)

    def test_idempotent_rerun(self, seeded_db):
        """Re-running pipeline produces same edge count."""
        run_pipeline(seeded_db, top_k=5, max_out=4, sim_min=0.0, conf_threshold=0.0)
        conn = phase1_db.get_connection(seeded_db)
        count1 = len(get_all_edges(conn))
        conn.close()

        run_pipeline(seeded_db, top_k=5, max_out=4, sim_min=0.0, conf_threshold=0.0)
        conn = phase1_db.get_connection(seeded_db)
        count2 = len(get_all_edges(conn))
        conn.close()

        assert count1 == count2

    def test_dry_run_no_persist(self, seeded_db):
        """Dry-run computes metrics but does not persist edges."""
        metrics = run_pipeline(
            seeded_db, top_k=5, max_out=4, sim_min=0.0,
            conf_threshold=0.0, dry_run=True,
        )
        conn = phase1_db.get_connection(seeded_db)
        rows = get_all_edges(conn)
        conn.close()
        # Dry run should leave table empty (clear_phase3_data not called
        # before insert in dry mode, but table may have been cleared)
        assert len(rows) == 0
