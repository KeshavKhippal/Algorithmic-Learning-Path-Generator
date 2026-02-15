#!/usr/bin/env python3
"""
Quick-check script for YouTube transcript fetching.

Usage::

    python scripts/check_youtube.py https://www.youtube.com/watch?v=kqtD5dpn9C8
    python scripts/check_youtube.py kqtD5dpn9C8

Prints:
- Extracted video ID
- Attempt logs
- Transcript preview (first 300 chars) on success
- Failure status + path to stored exception log on failure
"""

import argparse
import logging
import os
import re
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import extract_youtube_id, setup_logging
from src.fetchers.youtube_fetcher import fetch_transcript


def _resolve_input(raw: str) -> str:
    """Convert *raw* to a full YouTube URL if it looks like a bare video ID."""
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", raw):
        return f"https://www.youtube.com/watch?v={raw}"
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick-check YouTube transcript fetching.",
    )
    parser.add_argument(
        "url_or_id",
        help="YouTube URL or bare 11-character video ID.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("check_youtube")

    url = _resolve_input(args.url_or_id)
    video_id = extract_youtube_id(url)

    print(f"\n{'=' * 60}")
    print(f"  Input:    {args.url_or_id}")
    print(f"  URL:      {url}")
    print(f"  Video ID: {video_id or '⚠ PARSE FAILED'}")
    print(f"{'=' * 60}\n")

    if not video_id:
        print("❌ Could not extract a video ID. Check the URL format.")
        sys.exit(1)

    text, title, status = fetch_transcript(url)

    print(f"\n{'=' * 60}")
    print(f"  Status: {status}")
    print(f"  Title:  {title or '—'}")

    if status == "ok" and text:
        preview = text[:300]
        print(f"  Length: {len(text)} chars")
        print(f"\n  Transcript preview:\n  {'-' * 56}")
        print(f"  {preview}")
        if len(text) > 300:
            print(f"  … ({len(text) - 300} more chars)")
    else:
        # Check for failure log
        failures_dir = os.path.join(".", "data", "failures")
        if os.path.isdir(failures_dir):
            logs = sorted(
                f for f in os.listdir(failures_dir)
                if f.startswith(f"youtube_{video_id}_")
            )
            if logs:
                latest = os.path.join(failures_dir, logs[-1])
                print(f"  Failure log: {os.path.abspath(latest)}")
            else:
                print("  No failure log found.")
        else:
            print("  No failures directory exists yet.")

    print(f"{'=' * 60}\n")
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
