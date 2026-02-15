"""
Database module for the Algorithmic Learning Path Generator.

Phase 1: ``RawResources`` table + CRUD helpers.
Phase 2: ``ExtractedConcepts`` and ``CanonicalConcepts`` tables + helpers.
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# =========================================================================
# Schema constants
# =========================================================================

_CREATE_RAW_RESOURCES = """\
CREATE TABLE IF NOT EXISTS RawResources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url          TEXT    UNIQUE NOT NULL,
    content_type TEXT    CHECK(content_type IN ('article','youtube','unknown')),
    title        TEXT,
    raw_text     TEXT,
    status       TEXT    CHECK(status IN ('ok','no_content','no_transcript',
                                          'transcript_disabled','rate_limited',
                                          'video_unavailable','network_error',
                                          'failed','skipped')),
    extracted_at TIMESTAMP,
    notes        TEXT
);
"""

_CREATE_EXTRACTED_CONCEPTS = """\
CREATE TABLE IF NOT EXISTS ExtractedConcepts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id   INTEGER NOT NULL,
    concept       TEXT    NOT NULL,
    canonical_id  INTEGER NULL,
    sentence      TEXT,
    embedding     BLOB,
    created_at    TIMESTAMP,
    FOREIGN KEY (resource_id) REFERENCES RawResources(id)
);
"""

_CREATE_CANONICAL_CONCEPTS = """\
CREATE TABLE IF NOT EXISTS CanonicalConcepts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_concept   TEXT    UNIQUE NOT NULL,
    difficulty_score    REAL,
    difficulty_bucket   TEXT    CHECK(difficulty_bucket IN
                                      ('beginner','intermediate','advanced')),
    example_sentence    TEXT,
    resource_count      INTEGER,
    created_at          TIMESTAMP
);
"""


# =========================================================================
# Connection helper
# =========================================================================


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a new SQLite connection with WAL mode and row-factory enabled."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# =========================================================================
# Migration
# =========================================================================


def migrate_db(db_path: str) -> None:
    """Create (or verify) the ``RawResources`` table (Phase 1)."""
    conn = get_connection(db_path)
    try:
        conn.execute(_CREATE_RAW_RESOURCES)
        conn.commit()
        logger.info("Phase 1 migration OK at %s", os.path.abspath(db_path))
    finally:
        conn.close()


def migrate_phase2(db_path: str) -> None:
    """Create (or verify) Phase 2 tables: ``ExtractedConcepts``, ``CanonicalConcepts``."""
    conn = get_connection(db_path)
    try:
        conn.execute(_CREATE_EXTRACTED_CONCEPTS)
        conn.execute(_CREATE_CANONICAL_CONCEPTS)
        conn.commit()
        logger.info("Phase 2 migration OK at %s", os.path.abspath(db_path))
    finally:
        conn.close()


# =========================================================================
# Lock-retry helper
# =========================================================================

_SQLITE_LOCK_RETRIES = 5
_SQLITE_LOCK_BASE_DELAY = 0.1


def _retry_on_lock(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Wrap *fn* with SQLite-lock retry."""
    for attempt in range(1, _SQLITE_LOCK_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < _SQLITE_LOCK_RETRIES:
                delay = _SQLITE_LOCK_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "SQLite locked (attempt %d/%d) — retrying in %.2fs",
                    attempt, _SQLITE_LOCK_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                raise


# =========================================================================
# Phase 1 Write helpers
# =========================================================================


def insert_resource(
    conn: sqlite3.Connection,
    url: str,
    content_type: str,
    title: Optional[str],
    raw_text: Optional[str],
    status: str,
    notes: Optional[str] = None,
) -> int:
    """Insert a resource row idempotently (``INSERT OR IGNORE``)."""
    now = datetime.now(timezone.utc).isoformat()

    def _do_insert() -> int:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO RawResources
                (url, content_type, title, raw_text, status, extracted_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (url, content_type, title, raw_text, status, now, notes),
        )
        conn.commit()
        if cursor.rowcount == 1:
            return cursor.lastrowid  # type: ignore[return-value]
        row = conn.execute(
            "SELECT id FROM RawResources WHERE url = ?", (url,)
        ).fetchone()
        return row["id"] if row else -1

    return _retry_on_lock(_do_insert)


# =========================================================================
# Phase 1 Read helpers
# =========================================================================


def get_resource_by_url(
    conn: sqlite3.Connection, url: str
) -> Optional[Dict[str, Any]]:
    """Return the resource row for *url*, or ``None``."""
    row = conn.execute(
        "SELECT * FROM RawResources WHERE url = ?", (url,)
    ).fetchone()
    return dict(row) if row else None


def get_all_resources(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return every row in ``RawResources`` ordered by ``id``."""
    rows = conn.execute("SELECT * FROM RawResources ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def count_by_status(conn: sqlite3.Connection) -> Dict[str, int]:
    """Return a ``{status: count}`` mapping."""
    rows = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM RawResources GROUP BY status"
    ).fetchall()
    return {r["status"]: r["cnt"] for r in rows}


# =========================================================================
# Phase 2 Read helpers
# =========================================================================


def get_ok_resources(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all ``RawResources`` rows with ``status='ok'``."""
    rows = conn.execute(
        "SELECT * FROM RawResources WHERE status = 'ok' ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_extracted_concepts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all rows from ``ExtractedConcepts``."""
    rows = conn.execute(
        "SELECT * FROM ExtractedConcepts ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_canonical_concepts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all rows from ``CanonicalConcepts``."""
    rows = conn.execute(
        "SELECT * FROM CanonicalConcepts ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


# =========================================================================
# Phase 2 Write helpers
# =========================================================================


def clear_phase2_data(conn: sqlite3.Connection) -> None:
    """Wipe Phase 2 tables for idempotent re-runs.

    Deletes all rows from ``ExtractedConcepts`` and ``CanonicalConcepts``.
    """
    conn.execute("DELETE FROM ExtractedConcepts;")
    conn.execute("DELETE FROM CanonicalConcepts;")
    conn.commit()
    logger.info("Phase 2 tables cleared for idempotent re-run.")


def insert_extracted_concepts_batch(
    conn: sqlite3.Connection,
    rows: List[Dict[str, Any]],
) -> List[int]:
    """Insert a batch of extracted concepts. Returns list of row IDs.

    Each dict must have keys: ``resource_id``, ``concept``, ``sentence``,
    ``embedding`` (bytes).
    """
    now = datetime.now(timezone.utc).isoformat()
    ids: List[int] = []

    for row in rows:
        cursor = conn.execute(
            """
            INSERT INTO ExtractedConcepts
                (resource_id, concept, sentence, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (row["resource_id"], row["concept"], row["sentence"],
             row["embedding"], now),
        )
        ids.append(cursor.lastrowid)  # type: ignore[arg-type]

    conn.commit()
    return ids


def insert_canonical_concept(
    conn: sqlite3.Connection,
    canonical_concept: str,
    difficulty_score: float,
    difficulty_bucket: str,
    example_sentence: Optional[str],
    resource_count: int,
) -> int:
    """Insert a canonical concept (idempotent via UNIQUE constraint).

    Returns the row ID.
    """
    now = datetime.now(timezone.utc).isoformat()

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO CanonicalConcepts
            (canonical_concept, difficulty_score, difficulty_bucket,
             example_sentence, resource_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (canonical_concept, difficulty_score, difficulty_bucket,
         example_sentence, resource_count, now),
    )
    conn.commit()

    if cursor.rowcount == 1:
        return cursor.lastrowid  # type: ignore[return-value]

    row = conn.execute(
        "SELECT id FROM CanonicalConcepts WHERE canonical_concept = ?",
        (canonical_concept,),
    ).fetchone()
    return row["id"] if row else -1


def update_canonical_ids(
    conn: sqlite3.Connection,
    mapping: Dict[int, int],
) -> None:
    """Set ``canonical_id`` on ``ExtractedConcepts`` rows.

    Args:
        mapping: ``{extracted_concept_id: canonical_concept_id}``.
    """
    for ec_id, cc_id in mapping.items():
        conn.execute(
            "UPDATE ExtractedConcepts SET canonical_id = ? WHERE id = ?",
            (cc_id, ec_id),
        )
    conn.commit()
