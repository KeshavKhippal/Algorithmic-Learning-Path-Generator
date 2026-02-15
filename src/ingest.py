"""
Ingestion pipeline and CLI for the Algorithmic Learning Path Generator.

Usage::

    python -m src.ingest \\
        --input tests/sample_urls.json \\
        --db ./data/resources.db \\
        --batch-size 5

Exit code 0 on success.
"""

import argparse
import json
import logging
import os
import sys
from typing import List

from src import db
from src.fetchers import trafilatura_fetcher, youtube_fetcher
from src.models import IngestSummary
from src.utils import detect_content_type, retry_with_backoff, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-URL processing
# ---------------------------------------------------------------------------


def _ingest_one(
    url: str,
    conn,
    summary: IngestSummary,
    idx: int,
    batch_size: int,
) -> None:
    """Process one URL end-to-end and update *summary* in-place.

    This function **never** raises — every error is caught, logged,
    and translated into the appropriate status code.
    """
    url = url.strip()
    if not url:
        return

    logger.info("▶ [%d/%d] Processing: %s", idx, summary.total, url)

    # ---- idempotency check ----
    existing = db.get_resource_by_url(conn, url)
    if existing is not None:
        logger.info("  ⏭ Already ingested (id=%s), skipping.", existing["id"])
        summary.skipped += 1
        return

    content_type = detect_content_type(url)
    raw_text = None
    title = None
    status = "failed"
    notes = None

    # ---- fetch with retry ----
    try:
        if content_type == "youtube":
            raw_text, title, status = retry_with_backoff(
                youtube_fetcher.fetch_transcript,
                url,
                max_retries=3,
                base_delay=0.5,
                logger=logger,
            )
        elif content_type == "article":
            raw_text, title, status = retry_with_backoff(
                trafilatura_fetcher.fetch_article,
                url,
                max_retries=3,
                base_delay=0.5,
                logger=logger,
            )
        else:
            status = "skipped"
            notes = "Unknown URL scheme — cannot classify"
    except Exception as exc:
        logger.error(
            "  ✗ All retries exhausted for %s: %s", url, exc, exc_info=True
        )
        status = "failed"
        notes = str(exc)

    if status != "ok" and notes is None:
        notes = status  # persist the status reason as a note

    # ---- persist ----
    db.insert_resource(
        conn,
        url=url,
        content_type=content_type,
        title=title,
        raw_text=raw_text,
        status=status,
        notes=notes,
    )

    # ---- update summary counters ----
    _increment_summary(summary, status, url, notes)

    logger.info("  → status=%s", status)

    if idx % batch_size == 0:
        logger.info(
            "  📦 Batch checkpoint — %d/%d processed.", idx, summary.total
        )


def _increment_summary(
    summary: IngestSummary, status: str, url: str, notes: str | None
) -> None:
    """Increment the correct counter on *summary*."""
    if status == "ok":
        summary.succeeded += 1
    elif status == "no_content":
        summary.no_content += 1
    elif status == "no_transcript":
        summary.no_transcript += 1
    elif status == "transcript_disabled":
        summary.transcript_disabled += 1
    elif status == "rate_limited":
        summary.rate_limited += 1
    elif status == "video_unavailable":
        summary.video_unavailable += 1
    elif status == "network_error":
        summary.network_error += 1
    elif status == "skipped":
        summary.skipped += 1
    else:  # "failed" or anything unexpected
        summary.failed += 1
        summary.errors.append(f"{url}: {notes or 'unknown error'}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_urls(
    url_list: List[str],
    batch_size: int = 10,
    db_path: str = "./data/resources.db",
) -> IngestSummary:
    """Ingest a list of URLs into the local SQLite database.

    Args:
        url_list: URLs to process.
        batch_size: Log a checkpoint every *batch_size* URLs.
        db_path: Filesystem path to the SQLite database.

    Returns:
        An ``IngestSummary`` with aggregated statistics.
    """
    db.migrate_db(db_path)
    conn = db.get_connection(db_path)

    summary = IngestSummary(total=len(url_list))

    try:
        for idx, raw_url in enumerate(url_list, 1):
            _ingest_one(raw_url, conn, summary, idx, batch_size)
    finally:
        conn.close()

    # ---- write ingest_summary.json ----
    data_dir = os.path.dirname(os.path.abspath(db_path))
    summary_path = os.path.join(data_dir, "ingest_summary.json")
    os.makedirs(data_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary.model_dump_json(indent=2))
    logger.info("📄 Summary written to %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ingest",
        description="Ingest URLs into the Algorithmic Learning Path Generator database.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file containing a list of URLs.",
    )
    parser.add_argument(
        "--db",
        default="./data/resources.db",
        help="Path to the SQLite database (default: ./data/resources.db).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Checkpoint log interval (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """CLI main entry-point."""
    setup_logging()
    args = _parse_args(argv)

    logger.info("Loading URLs from %s", args.input)
    with open(args.input, "r", encoding="utf-8") as fh:
        url_list: List[str] = json.load(fh)

    if not isinstance(url_list, list):
        logger.error("Expected a JSON array of URL strings in %s", args.input)
        sys.exit(1)

    summary = ingest_urls(
        url_list=url_list,
        batch_size=args.batch_size,
        db_path=args.db,
    )

    logger.info(
        "✅ Ingestion complete — total=%d succeeded=%d failed=%d "
        "no_content=%d no_transcript=%d transcript_disabled=%d "
        "rate_limited=%d video_unavailable=%d network_error=%d skipped=%d",
        summary.total,
        summary.succeeded,
        summary.failed,
        summary.no_content,
        summary.no_transcript,
        summary.transcript_disabled,
        summary.rate_limited,
        summary.video_unavailable,
        summary.network_error,
        summary.skipped,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
