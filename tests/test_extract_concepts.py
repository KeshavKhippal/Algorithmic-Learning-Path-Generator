"""
pytest test suite for Phase 2: Concept Extraction + Canonicalization.

All tests are mocked — no internet or GPU required.
Run with::

    pytest tests/test_extract_concepts.py -v

Integration tests that load real models are marked with
``@pytest.mark.integration`` and skipped by default.
"""

import json
import os
import sqlite3
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db
from src.extractors.spacy_extractor import (
    CandidateResult,
    extract_candidates,
    extract_all,
    _normalise,
    _is_valid_candidate,
)
from src.embeddings.embedder import embed_texts
from src.canonicalization.clusterer import (
    canonicalize,
    _choose_canonical,
    _score_difficulty,
    _build_candidate_index,
    CanonicalCluster,
)
from src.utils import embedding_to_blob, blob_to_embedding, set_global_seed
from src.extract_concepts import run_pipeline, _build_summary


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def tmp_db(tmp_path):
    """Return a DB path inside a temporary directory."""
    return str(tmp_path / "test_phase2.db")


@pytest.fixture()
def sample_resources():
    """Load sample Phase 2 resources."""
    path = os.path.join(os.path.dirname(__file__), "sample_phase2_resources.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture()
def seeded_db(tmp_db, sample_resources):
    """Create a DB pre-seeded with Phase 1 + Phase 2 tables and sample resources."""
    db.migrate_db(tmp_db)
    db.migrate_phase2(tmp_db)
    conn = db.get_connection(tmp_db)
    for res in sample_resources:
        db.insert_resource(
            conn,
            url=res["url"],
            content_type=res["content_type"],
            title=res.get("title"),
            raw_text=res.get("raw_text"),
            status=res["status"],
        )
    conn.close()
    return tmp_db


# =========================================================================
# Mock embedding helper
# =========================================================================


def _mock_embed_texts(texts, batch_size=64, model=None):
    """Return deterministic mock embeddings of shape (N, 384)."""
    np.random.seed(42)
    n = len(texts)
    if n == 0:
        return np.empty((0, 384), dtype=np.float32)
    emb = np.random.randn(n, 384).astype(np.float32)
    # Normalise
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return emb


def _mock_embed_texts_similar_react(texts, batch_size=64, model=None):
    """Return embeddings where 'reactjs' and 'reactjs' are near-identical."""
    np.random.seed(42)
    n = len(texts)
    emb = np.random.randn(n, 384).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms

    # Make semantically similar concepts have near-identical embeddings
    text_list = list(texts)
    for i, t in enumerate(text_list):
        if "react" in t.lower():
            # Set all react-related concepts to the same base vector
            base = emb[i].copy()
            for j, t2 in enumerate(text_list):
                if "react" in t2.lower() and j != i:
                    emb[j] = base + np.random.randn(384).astype(np.float32) * 0.01
            break

    # Re-normalise
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return emb


# =========================================================================
# Test: Candidate Extraction
# =========================================================================


class TestCandidateExtraction:
    """Tests for NLTK-based concept candidate extraction."""

    def test_basic_extraction(self):
        """Extract noun-phrase candidates from simple text."""
        text = (
            "Python is a popular programming language. "
            "It supports object-oriented programming."
        )
        result = extract_candidates(resource_id=1, raw_text=text)
        assert isinstance(result, CandidateResult)
        assert result.resource_id == 1
        assert len(result.candidates) > 0

        # Should find "python" and "programming language" or similar
        concepts = [c for c, _s in result.candidates]
        assert any("python" in c for c in concepts)

    def test_deduplication_per_resource(self):
        """Same concept mentioned twice should appear only once."""
        text = (
            "Python is great. Python is used in data science. "
            "Python is also used in web development."
        )
        result = extract_candidates(resource_id=1, raw_text=text)
        concepts = [c for c, _s in result.candidates]
        python_count = sum(1 for c in concepts if c == "python")
        assert python_count <= 1

    def test_normalisation(self):
        """Verify normalisation strips punctuation and lowercases."""
        assert _normalise("  Machine   Learning  ") == "machine learning"
        assert _normalise("Python!") == "python"
        assert _normalise("C++") == "c"

    def test_validity_filter(self):
        """Short/numeric/stopword candidates are rejected."""
        assert _is_valid_candidate("x") is False
        assert _is_valid_candidate("123") is False
        assert _is_valid_candidate("the") is False
        assert _is_valid_candidate("machine learning") is True

    def test_sentence_context_preserved(self):
        """Each candidate should have its source sentence."""
        text = "Python supports decorators. Decorators modify functions."
        result = extract_candidates(resource_id=1, raw_text=text)
        for concept, sentence in result.candidates:
            assert len(sentence) > 0
            assert isinstance(sentence, str)

    def test_extract_all_multiple_resources(self, sample_resources):
        """extract_all handles multiple resources."""
        results = extract_all(sample_resources)
        assert len(results) == 3
        for cr in results:
            assert isinstance(cr, CandidateResult)
            assert len(cr.candidates) > 0

    def test_empty_text_skipped(self):
        """Resources with empty text produce no candidates."""
        resources = [{"id": 1, "raw_text": ""}]
        results = extract_all(resources)
        assert len(results) == 0


# =========================================================================
# Test: Embedding and Serialisation
# =========================================================================


class TestEmbeddingAndSerialization:
    """Tests for embedding shape, dtype, and BLOB serialisation."""

    def test_embedding_shape_and_dtype(self):
        """embed_texts produces (N, 384) float32 arrays."""
        texts = ["python programming", "machine learning"]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384).astype(
            np.float32
        )
        result = embed_texts(texts, model=mock_model)
        assert result.shape == (2, 384)
        assert result.dtype == np.float32

    def test_empty_input(self):
        """Empty input returns (0, 384) array."""
        result = embed_texts([])
        assert result.shape == (0, 384)

    def test_blob_round_trip(self):
        """embedding_to_blob → blob_to_embedding preserves data."""
        original = np.random.randn(384).astype(np.float32)
        blob = embedding_to_blob(original)
        restored = blob_to_embedding(blob, dim=384)
        np.testing.assert_array_almost_equal(original, restored)

    def test_blob_is_bytes(self):
        """embedding_to_blob returns bytes."""
        arr = np.zeros(384, dtype=np.float32)
        blob = embedding_to_blob(arr)
        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4  # float32 = 4 bytes


# =========================================================================
# Test: Clustering and Canonicalization
# =========================================================================


class TestClusteringAndCanonicalization:
    """Tests for the canonicalization pipeline."""

    def test_canonical_label_selection(self):
        """Shortest by token count, tie-break by frequency."""
        from collections import Counter

        members = ["machine learning algorithm", "ml algorithm", "ml"]
        freq = Counter({"ml": 5, "ml algorithm": 3, "machine learning algorithm": 1})
        label = _choose_canonical(members, freq)
        assert label == "ml"

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts_similar_react)
    def test_similar_concepts_same_cluster(self, _embed):
        """Semantically similar strings should cluster together."""
        # Create two results with "reactjs" and "reactjs" (same text)
        results = [
            CandidateResult(resource_id=1, candidates=[
                ("reactjs", "ReactJS is a library."),
                ("virtual dom", "Uses virtual DOM."),
            ]),
            CandidateResult(resource_id=2, candidates=[
                ("reactjs", "ReactJS is popular."),
                ("javascript", "JavaScript is versatile."),
            ]),
        ]
        clusters, index, embeddings = canonicalize(results, cluster_threshold=0.85)
        assert len(clusters) > 0

        # Find the cluster containing "reactjs"
        reactjs_clusters = [
            cl for cl in clusters if "reactjs" in cl.members
        ]
        assert len(reactjs_clusters) >= 1

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_single_candidate_clusters(self, _embed):
        """Single candidate produces a single cluster."""
        results = [
            CandidateResult(resource_id=1, candidates=[
                ("python", "Python is great."),
            ]),
        ]
        clusters, _, _ = canonicalize(results, cluster_threshold=0.85)
        assert len(clusters) == 1
        assert clusters[0].canonical_label == "python"

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_resource_count_tracked(self, _embed):
        """resource_ids should track which resources contribute."""
        results = [
            CandidateResult(resource_id=1, candidates=[
                ("python", "Python is great."),
            ]),
            CandidateResult(resource_id=2, candidates=[
                ("python", "Python for ML."),
            ]),
        ]
        clusters, _, _ = canonicalize(results, cluster_threshold=0.85)
        python_cl = [c for c in clusters if c.canonical_label == "python"]
        assert len(python_cl) == 1
        assert python_cl[0].resource_ids == {1, 2}


# =========================================================================
# Test: Difficulty Score
# =========================================================================


class TestDifficultyScoreRange:
    """Tests for difficulty scoring and bucket assignment."""

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_score_in_range(self, _embed):
        """All difficulty scores must be in [0, 1]."""
        results = [
            CandidateResult(resource_id=1, candidates=[
                ("python", "Python is a language."),
                ("machine learning", "Machine learning uses data."),
                ("neural networks", "Neural networks have layers with complex architecture."),
            ]),
        ]
        clusters, _, _ = canonicalize(results, cluster_threshold=0.85)
        for cl in clusters:
            assert 0.0 <= cl.difficulty_score <= 1.0, (
                f"{cl.canonical_label}: score={cl.difficulty_score}"
            )

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_bucket_assignment(self, _embed):
        """Difficulty buckets must be one of beginner/intermediate/advanced."""
        results = [
            CandidateResult(resource_id=1, candidates=[
                ("python", "Python is a language."),
                ("decorators", "Python decorators modify functions with complex metaprogramming patterns."),
            ]),
        ]
        clusters, _, _ = canonicalize(results, cluster_threshold=0.85)
        valid_buckets = {"beginner", "intermediate", "advanced"}
        for cl in clusters:
            assert cl.difficulty_bucket in valid_buckets


# =========================================================================
# Test: Determinism
# =========================================================================


class TestDeterminism:
    """Verify that the pipeline produces identical results on re-run."""

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_deterministic_output(self, _embed):
        """Running canonicalize twice produces identical clusters."""
        results = [
            CandidateResult(resource_id=1, candidates=[
                ("python", "Python is great."),
                ("machine learning", "ML is powerful."),
            ]),
            CandidateResult(resource_id=2, candidates=[
                ("data science", "Data science uses Python."),
                ("python", "Python for data."),
            ]),
        ]

        set_global_seed(42)
        clusters1, _, _ = canonicalize(results, cluster_threshold=0.85)

        set_global_seed(42)
        clusters2, _, _ = canonicalize(results, cluster_threshold=0.85)

        labels1 = sorted(cl.canonical_label for cl in clusters1)
        labels2 = sorted(cl.canonical_label for cl in clusters2)
        assert labels1 == labels2

        scores1 = sorted(cl.difficulty_score for cl in clusters1)
        scores2 = sorted(cl.difficulty_score for cl in clusters2)
        assert scores1 == scores2


# =========================================================================
# Test: Database Integration
# =========================================================================


class TestDatabaseIntegration:
    """Integration tests using a real temp DB (no network)."""

    def test_phase2_tables_created(self, tmp_db):
        """Phase 2 migration creates both tables."""
        db.migrate_db(tmp_db)
        db.migrate_phase2(tmp_db)
        conn = db.get_connection(tmp_db)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        conn.close()

        assert "ExtractedConcepts" in table_names
        assert "CanonicalConcepts" in table_names

    def test_clear_phase2_idempotent(self, seeded_db):
        """clear_phase2_data should not error on empty tables."""
        conn = db.get_connection(seeded_db)
        db.clear_phase2_data(conn)  # first clear
        db.clear_phase2_data(conn)  # second clear — should not error
        conn.close()

    def test_canonical_unique_constraint(self, seeded_db):
        """Inserting same canonical_concept twice should not duplicate."""
        conn = db.get_connection(seeded_db)
        id1 = db.insert_canonical_concept(
            conn, "python", 0.3, "beginner", "Python is great.", 5
        )
        id2 = db.insert_canonical_concept(
            conn, "python", 0.3, "beginner", "Python is great.", 5
        )
        rows = db.get_canonical_concepts(conn)
        conn.close()

        python_rows = [r for r in rows if r["canonical_concept"] == "python"]
        assert len(python_rows) == 1
        assert id1 == id2

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_full_pipeline_mock(self, _embed, seeded_db):
        """End-to-end pipeline with mocked embeddings produces correct DB state."""
        summary = run_pipeline(db_path=seeded_db, batch_size=64)

        assert summary.resources_processed == 3
        assert summary.candidate_concepts > 0
        assert summary.canonical_concepts > 0

        conn = db.get_connection(seeded_db)
        ec_rows = db.get_extracted_concepts(conn)
        cc_rows = db.get_canonical_concepts(conn)
        conn.close()

        assert len(ec_rows) > 0
        assert len(cc_rows) > 0
        assert len(cc_rows) == summary.canonical_concepts

        # All extracted concepts should have canonical_id set
        for row in ec_rows:
            assert row["canonical_id"] is not None

        # All canonical concepts should have valid difficulty
        for row in cc_rows:
            assert 0.0 <= row["difficulty_score"] <= 1.0
            assert row["difficulty_bucket"] in ("beginner", "intermediate", "advanced")
            assert row["resource_count"] >= 1

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_summary_json_written(self, _embed, seeded_db):
        """phase2_summary.json should be written next to the DB."""
        run_pipeline(db_path=seeded_db)

        summary_path = os.path.join(
            os.path.dirname(seeded_db), "phase2_summary.json"
        )
        assert os.path.isfile(summary_path)

        with open(summary_path) as fh:
            data = json.load(fh)

        assert data["resources_processed"] == 3
        assert data["canonical_concepts"] > 0
        assert "difficulty_distribution" in data

    @patch("src.canonicalization.clusterer.embed_texts", side_effect=_mock_embed_texts)
    def test_idempotent_rerun(self, _embed, seeded_db):
        """Re-running the pipeline should not duplicate rows."""
        run_pipeline(db_path=seeded_db)
        conn = db.get_connection(seeded_db)
        cc1 = len(db.get_canonical_concepts(conn))
        ec1 = len(db.get_extracted_concepts(conn))
        conn.close()

        run_pipeline(db_path=seeded_db)
        conn = db.get_connection(seeded_db)
        cc2 = len(db.get_canonical_concepts(conn))
        ec2 = len(db.get_extracted_concepts(conn))
        conn.close()

        assert cc1 == cc2
        assert ec1 == ec2
