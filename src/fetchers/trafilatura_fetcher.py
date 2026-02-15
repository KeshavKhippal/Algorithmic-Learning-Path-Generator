"""
Trafilatura-based fetcher for web articles.

Extracts main body text and title from arbitrary web pages using
``trafilatura.fetch_url`` and ``trafilatura.extract``.
"""

import logging
from typing import Tuple, Optional

import trafilatura

logger = logging.getLogger(__name__)


def fetch_article(url: str) -> Tuple[Optional[str], Optional[str], str]:
    """Fetch and extract content from a web article.

    Args:
        url: The article URL to fetch.

    Returns:
        A tuple of ``(raw_text, title, status)`` where *status* is one of
        ``'ok'``, ``'no_content'``, ``'rate_limited'``, or ``'failed'``.

    Raises:
        No exceptions are raised; all errors are caught internally and
        reflected in the returned *status*.
    """
    try:
        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            logger.warning("trafilatura.fetch_url returned None for %s", url)
            return None, None, "no_content"

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            favor_precision=True,
            output_format="txt",
        )

        if not text:
            logger.warning("trafilatura.extract returned empty for %s", url)
            return None, None, "no_content"

        # Attempt to pull the title from metadata
        title: Optional[str] = None
        try:
            metadata = trafilatura.extract_metadata(downloaded)
            if metadata and metadata.title:
                title = metadata.title
        except Exception:
            pass  # title is nice-to-have

        if not title:
            # Derive a title from the URL path as a fallback
            title = url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ").title()

        return text, title, "ok"

    except Exception as exc:
        # Heuristic: treat 429-related messages as rate limiting
        msg = str(exc).lower()
        if "429" in msg or "rate" in msg:
            logger.error("Rate limited while fetching %s: %s", url, exc)
            return None, None, "rate_limited"

        logger.error("Failed to fetch article %s: %s", url, exc, exc_info=True)
        return None, None, "failed"
