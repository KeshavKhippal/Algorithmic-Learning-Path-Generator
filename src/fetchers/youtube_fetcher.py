"""
YouTube transcript fetcher for ``youtube-transcript-api`` **v1.x**.

The v1.x library changed its surface significantly:
- ``YouTubeTranscriptApi`` must be **instantiated**.
- ``.fetch(video_id)`` returns a ``FetchedTranscript`` with ``.snippets``
  (list of ``FetchedTranscriptSnippet`` objects with ``.text``).
- ``.list(video_id)`` returns a ``TranscriptList``.

This module provides:
- ``FetchTranscriptError`` — typed exception carrying a status code.
- ``fetch_transcript(url)`` — public entry-point for the ingestion pipeline.
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Optional, Tuple

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.utils import extract_youtube_id

logger = logging.getLogger(__name__)

# Singleton API instance (thread-safe for reads)
_api = YouTubeTranscriptApi()

# Statuses that indicate transient errors worth retrying
_TRANSIENT_STATUSES = frozenset({"rate_limited", "network_error"})


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class FetchTranscriptError(Exception):
    """Raised when transcript fetching fails after all retries.

    Attributes:
        status: One of ``'no_transcript'``, ``'transcript_disabled'``,
                ``'rate_limited'``, ``'video_unavailable'``,
                ``'network_error'``, ``'failed'``.
        original: The original underlying exception (may be ``None``).
    """

    def __init__(self, status: str, original: Optional[Exception] = None) -> None:
        self.status = status
        self.original = original
        super().__init__(f"status={status}: {original}")


# ---------------------------------------------------------------------------
# Failure log writer
# ---------------------------------------------------------------------------

_FAILURES_DIR = os.path.join(".", "data", "failures")


def _write_failure_log(video_id: str, exc: Exception) -> str:
    """Persist a full exception traceback to disk.

    Returns:
        The path to the written log file.
    """
    os.makedirs(_FAILURES_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"youtube_{video_id}_{ts}.log"
    filepath = os.path.join(_FAILURES_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(f"video_id: {video_id}\n")
        fh.write(f"timestamp: {ts}\n")
        fh.write(f"exception_type: {type(exc).__name__}\n")
        fh.write(f"exception_msg: {exc}\n\n")
        fh.write(traceback.format_exc())

    return filepath


# ---------------------------------------------------------------------------
# Internal: classify exceptions → status
# ---------------------------------------------------------------------------


def _classify_exception(exc: Exception) -> str:
    """Map a library / network exception to a status string."""
    if isinstance(exc, TranscriptsDisabled):
        return "transcript_disabled"

    if isinstance(exc, NoTranscriptFound):
        return "no_transcript"

    if isinstance(exc, VideoUnavailable):
        return "video_unavailable"

    msg = str(exc).lower()

    # HTTP 429 / too-many-requests
    if "429" in msg or "too many" in msg or "rate" in msg:
        return "rate_limited"

    # Network / DNS / timeout
    if any(kw in msg for kw in ("timeout", "timed out", "dns", "connectionerror",
                                 "connection refused", "network", "unreachable")):
        return "network_error"

    return "failed"


# ---------------------------------------------------------------------------
# Internal: single-attempt raw fetch
# ---------------------------------------------------------------------------


def _raw_fetch(video_id: str) -> str:
    """Attempt to fetch the transcript text for *video_id*.

    Returns:
        Concatenated transcript text.

    Raises:
        Any exception from ``youtube-transcript-api``.
    """
    result = _api.fetch(video_id)
    # result is a FetchedTranscript; iterate .snippets
    return " ".join(snippet.text for snippet in result.snippets)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_transcript(url: str) -> Tuple[Optional[str], Optional[str], str]:
    """Fetch a YouTube video transcript with retry for transient errors.

    Retries up to 3 times with exponential backoff (0.5 s → 1 s → 2 s) for
    transient statuses (``rate_limited``, ``network_error``).

    Args:
        url: YouTube video URL.

    Returns:
        A tuple ``(raw_text, title, status)`` where *status* is one of:
        ``'ok'``, ``'no_transcript'``, ``'transcript_disabled'``,
        ``'rate_limited'``, ``'video_unavailable'``, ``'network_error'``,
        ``'failed'``.
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        logger.error(
            "video_id_parse_failed | url=%s", url
        )
        return None, None, "failed"

    title = f"YouTube Video: {video_id}"
    max_retries = 3
    base_delay = 0.5
    last_status = "failed"
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        t0 = time.monotonic()
        try:
            text = _raw_fetch(video_id)
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "fetch_ok | url=%s | video_id=%s | attempt=%d | "
                "result_status=ok | elapsed_ms=%.0f",
                url, video_id, attempt, elapsed_ms,
            )
            return text, title, "ok"

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            status = _classify_exception(exc)
            last_status = status
            last_exc = exc

            logger.warning(
                "fetch_attempt_failed | url=%s | video_id=%s | attempt=%d/%d | "
                "result_status=%s | elapsed_ms=%.0f | exception=%s",
                url, video_id, attempt, max_retries, status, elapsed_ms,
                repr(exc),
            )

            # Retry only for transient errors
            if status in _TRANSIENT_STATUSES and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.info("  ↻ retrying in %.1fs …", delay)
                time.sleep(delay)
                continue

            # Non-transient or last attempt — log full trace and bail
            log_path = _write_failure_log(video_id, exc)
            logger.error(
                "fetch_final_failure | url=%s | video_id=%s | "
                "result_status=%s | failure_log=%s",
                url, video_id, status, log_path,
                exc_info=True,
            )
            return None, None, status

    # Fallback (should not be reached)
    if last_exc:
        _write_failure_log(video_id, last_exc)
    return None, None, last_status
