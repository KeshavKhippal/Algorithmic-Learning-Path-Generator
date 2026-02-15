"""
Utility helpers for the Algorithmic Learning Path Generator.

Provides:
- Structured logging configuration with timestamps.
- Retry with exponential back-off.
- URL content-type detection.
- Global seed setting for determinism.
- Embedding ↔ BLOB serialisation helpers.
"""

import logging
import os
import re
import time
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import parse_qs, urlparse

import numpy as np

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%dT%H:%M:%S%z",
) -> None:
    """Configure the root logger with timestamped structured output."""
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for numpy and Python hash for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Embedding serialisation
# ---------------------------------------------------------------------------


def embedding_to_blob(arr: np.ndarray) -> bytes:
    """Serialise a numpy float32 array to raw bytes for SQLite BLOB."""
    return arr.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes, dim: int = 384) -> np.ndarray:
    """Deserialise a SQLite BLOB back to a numpy float32 array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Retry with exponential back-off
# ---------------------------------------------------------------------------


def retry_with_backoff(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 0.5,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> T:
    """Call *fn* with retry and exponential back-off on exception."""
    if logger is None:
        logger = logging.getLogger(__name__)

    last_exc: BaseException = RuntimeError("unreachable")
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                attempt,
                max_retries,
                fn.__name__,
                exc,
                delay,
            )
            time.sleep(delay)

    raise last_exc


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

_YOUTUBE_PATTERNS = [
    re.compile(r"(www\.|m\.)?youtube\.com/watch"),
    re.compile(r"(www\.)?youtu\.be/"),
    re.compile(r"(www\.|m\.)?youtube\.com/embed/"),
]


def is_youtube_url(url: str) -> bool:
    """Return ``True`` if *url* looks like a YouTube video URL."""
    return any(p.search(url) for p in _YOUTUBE_PATTERNS)


def extract_youtube_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID from *url*."""
    _logger = logging.getLogger(__name__)

    if "youtu.be/" in url:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)
        _logger.debug("youtu.be pattern matched but no 11-char ID found in %s", url)
        return None

    if "/embed/" in url:
        match = re.search(r"/embed/([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)
        _logger.debug("/embed/ pattern matched but no 11-char ID found in %s", url)
        return None

    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        vid = query_params.get("v", [None])[0]
        if vid and re.fullmatch(r"[a-zA-Z0-9_-]{11}", vid):
            return vid
        _logger.debug("youtube.com matched but v= param missing/invalid in %s", url)
        return None

    _logger.debug("URL does not match any known YouTube pattern: %s", url)
    return None


def detect_content_type(url: str) -> str:
    """Classify a URL as ``'youtube'``, ``'article'``, or ``'unknown'``."""
    if is_youtube_url(url):
        return "youtube"
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return "article"
    return "unknown"
