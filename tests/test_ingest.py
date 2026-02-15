"""
pytest test suite for Phase 1: Automated Resource Ingestion.

All external network calls are mocked — no internet required.
Run with::

    pytest tests/test_ingest.py -v

Integration tests that require network access are marked with
``@pytest.mark.integration`` and skipped in CI.
"""

import json
import os
import sqlite3
from unittest.mock import patch, MagicMock

import pytest

# Ensure the project root is importable
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db
from src.ingest import ingest_urls
from src.utils import detect_content_type, is_youtube_url, extract_youtube_id
from src.fetchers.youtube_fetcher import fetch_transcript, FetchTranscriptError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    """Return a database path inside a temporary directory."""
    return str(tmp_path / "test_resources.db")


@pytest.fixture()
def sample_urls():
    """Load the sample URL list shipped with the repo."""
    path = os.path.join(os.path.dirname(__file__), "sample_urls.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

_MOCK_ARTICLE_TEXT = "This is a mock article body with enough words."
_MOCK_TRANSCRIPT_TEXT = "hello welcome to this tutorial on algorithms"


def _mock_fetch_article(url):
    """Simulate trafilatura_fetcher.fetch_article."""
    return _MOCK_ARTICLE_TEXT, "Mock Article Title", "ok"


def _mock_fetch_transcript(url):
    """Simulate youtube_fetcher.fetch_transcript."""
    return _MOCK_TRANSCRIPT_TEXT, "YouTube Video: abc123", "ok"


def _mock_fetch_no_transcript(url):
    """Simulate NoTranscriptFound."""
    return None, None, "no_transcript"


def _mock_fetch_transcript_disabled(url):
    """Simulate TranscriptsDisabled."""
    return None, None, "transcript_disabled"


def _mock_fetch_no_content(url):
    """Simulate trafilatura returning None."""
    return None, None, "no_content"


def _mock_fetch_video_unavailable(url):
    """Simulate VideoUnavailable."""
    return None, None, "video_unavailable"


def _mock_fetch_network_error(url):
    """Simulate a network error."""
    return None, None, "network_error"


# ---------------------------------------------------------------------------
# Tests: URL helpers
# ---------------------------------------------------------------------------


class TestUtilHelpers:
    """Quick sanity checks for URL utilities."""

    def test_detect_youtube(self):
        assert detect_content_type("https://www.youtube.com/watch?v=abc1234_xyz") == "youtube"

    def test_detect_article(self):
        assert detect_content_type("https://realpython.com/article") == "article"

    def test_is_youtube_url_short(self):
        assert is_youtube_url("https://youtu.be/abc1234_xyz")

    def test_is_youtube_url_mobile(self):
        assert is_youtube_url("https://m.youtube.com/watch?v=abc1234_xyz")

    def test_extract_youtube_id_standard(self):
        assert extract_youtube_id("https://www.youtube.com/watch?v=kqtD5dpn9C8") == "kqtD5dpn9C8"

    def test_extract_youtube_id_short(self):
        assert extract_youtube_id("https://youtu.be/kqtD5dpn9C8") == "kqtD5dpn9C8"


class TestExtractYoutubeIdVarieties:
    """Comprehensive YouTube ID parsing tests for all URL forms."""

    def test_standard_watch(self):
        assert extract_youtube_id("https://www.youtube.com/watch?v=kqtD5dpn9C8") == "kqtD5dpn9C8"

    def test_standard_watch_with_extra_params(self):
        assert extract_youtube_id(
            "https://www.youtube.com/watch?v=kqtD5dpn9C8&list=PLxxx&t=120"
        ) == "kqtD5dpn9C8"

    def test_short_url(self):
        assert extract_youtube_id("https://youtu.be/kqtD5dpn9C8") == "kqtD5dpn9C8"

    def test_short_url_with_time(self):
        assert extract_youtube_id("https://youtu.be/kqtD5dpn9C8?t=42") == "kqtD5dpn9C8"

    def test_embed_url(self):
        assert extract_youtube_id("https://www.youtube.com/embed/kqtD5dpn9C8") == "kqtD5dpn9C8"

    def test_embed_url_with_params(self):
        assert extract_youtube_id(
            "https://www.youtube.com/embed/kqtD5dpn9C8?autoplay=1&rel=0"
        ) == "kqtD5dpn9C8"

    def test_mobile_url(self):
        assert extract_youtube_id("https://m.youtube.com/watch?v=kqtD5dpn9C8") == "kqtD5dpn9C8"

    def test_mobile_url_with_feature(self):
        assert extract_youtube_id(
            "https://m.youtube.com/watch?v=kqtD5dpn9C8&feature=share"
        ) == "kqtD5dpn9C8"

    def test_no_scheme_returns_none(self):
        assert extract_youtube_id("not-a-url") is None

    def test_unrelated_url_returns_none(self):
        assert extract_youtube_id("https://example.com/watch?v=abc") is None


# ---------------------------------------------------------------------------
# Tests: YouTube fetcher
# ---------------------------------------------------------------------------


class TestFetchYoutubeTranscript:
    """Tests for the YouTube fetcher using monkeypatching."""

    def test_fetch_youtube_transcript_no_transcript(self):
        """Library's NoTranscriptFound → status='no_transcript'."""
        from youtube_transcript_api._errors import NoTranscriptFound

        with patch("src.fetchers.youtube_fetcher._api") as mock_api:
            mock_api.fetch.side_effect = NoTranscriptFound(
                "kqtD5dpn9C8", ["de"], ["de"]
            )
            text, title, status = fetch_transcript(
                "https://www.youtube.com/watch?v=kqtD5dpn9C8"
            )

        assert status == "no_transcript"
        assert text is None

    def test_fetch_youtube_transcript_disabled(self):
        """Library's TranscriptsDisabled → status='transcript_disabled'."""
        from youtube_transcript_api._errors import TranscriptsDisabled

        with patch("src.fetchers.youtube_fetcher._api") as mock_api:
            mock_api.fetch.side_effect = TranscriptsDisabled("kqtD5dpn9C8")
            text, title, status = fetch_transcript(
                "https://www.youtube.com/watch?v=kqtD5dpn9C8"
            )

        assert status == "transcript_disabled"
        assert text is None

    def test_fetch_youtube_transcript_video_unavailable(self):
        """Library's VideoUnavailable → status='video_unavailable'."""
        from youtube_transcript_api._errors import VideoUnavailable

        with patch("src.fetchers.youtube_fetcher._api") as mock_api:
            mock_api.fetch.side_effect = VideoUnavailable("kqtD5dpn9C8")
            text, title, status = fetch_transcript(
                "https://www.youtube.com/watch?v=kqtD5dpn9C8"
            )

        assert status == "video_unavailable"
        assert text is None

    def test_fetch_youtube_transcript_success_mock(self):
        """Successful fetch returns text and status='ok'."""
        mock_transcript = MagicMock()
        snippet1 = MagicMock()
        snippet1.text = "hello world"
        snippet2 = MagicMock()
        snippet2.text = "this is a test"
        mock_transcript.snippets = [snippet1, snippet2]

        with patch("src.fetchers.youtube_fetcher._api") as mock_api:
            mock_api.fetch.return_value = mock_transcript
            text, title, status = fetch_transcript(
                "https://www.youtube.com/watch?v=kqtD5dpn9C8"
            )

        assert status == "ok"
        assert text == "hello world this is a test"
        assert title == "YouTube Video: kqtD5dpn9C8"

    def test_parse_failure_returns_failed(self):
        """Unparseable URL → status='failed'."""
        text, title, status = fetch_transcript("not-a-youtube-url")
        assert status == "failed"
        assert text is None


# ---------------------------------------------------------------------------
# Tests: Integration with mocked fetchers
# ---------------------------------------------------------------------------


class TestIngestURLs:
    """Integration-style tests using mocked fetchers."""

    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript)
    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    def test_ingest_inserts_rows(self, _art, _yt, tmp_db, sample_urls):
        """Ingesting 3 URLs must produce 3 rows in the database."""
        summary = ingest_urls(sample_urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert len(rows) == 3
        assert summary.total == 3

    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript)
    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    def test_status_field_populated(self, _art, _yt, tmp_db, sample_urls):
        """Every row must have a non-null ``status`` value."""
        ingest_urls(sample_urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        valid_statuses = {"ok", "no_content", "no_transcript", "transcript_disabled",
                          "rate_limited", "video_unavailable", "network_error",
                          "failed", "skipped"}
        for row in rows:
            assert row["status"] is not None
            assert row["status"] in valid_statuses

    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript)
    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    def test_raw_text_not_null_when_ok(self, _art, _yt, tmp_db, sample_urls):
        """When ``status='ok'``, ``raw_text`` must not be NULL."""
        ingest_urls(sample_urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        ok_rows = [r for r in rows if r["status"] == "ok"]
        assert len(ok_rows) > 0, "Expected at least one row with status 'ok'"
        for row in ok_rows:
            assert row["raw_text"] is not None
            assert len(row["raw_text"]) > 0

    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript)
    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    def test_idempotency(self, _art, _yt, tmp_db):
        """Ingesting the same URL twice must not create a duplicate row."""
        urls = ["https://realpython.com/python-f-strings/"]
        ingest_urls(urls, db_path=tmp_db)
        summary2 = ingest_urls(urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert len(rows) == 1
        assert summary2.skipped == 1

    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_no_transcript)
    def test_no_transcript_status(self, _yt, _art, tmp_db):
        """A YouTube URL with no transcript must get ``status='no_transcript'``."""
        urls = ["https://www.youtube.com/watch?v=missing12345"]
        ingest_urls(urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert rows[0]["status"] == "no_transcript"

    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript_disabled)
    def test_transcript_disabled_status(self, _yt, _art, tmp_db):
        """Disabled transcripts must get ``status='transcript_disabled'``."""
        urls = ["https://www.youtube.com/watch?v=disabled1234"]
        summary = ingest_urls(urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert rows[0]["status"] == "transcript_disabled"
        assert summary.transcript_disabled == 1

    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_video_unavailable)
    def test_video_unavailable_status(self, _yt, _art, tmp_db):
        """Unavailable video must get ``status='video_unavailable'``."""
        urls = ["https://www.youtube.com/watch?v=unavailabl1"]
        summary = ingest_urls(urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert rows[0]["status"] == "video_unavailable"
        assert summary.video_unavailable == 1

    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_network_error)
    def test_network_error_status(self, _yt, _art, tmp_db):
        """Network errors must get ``status='network_error'``."""
        urls = ["https://www.youtube.com/watch?v=neterror1234"]
        summary = ingest_urls(urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert rows[0]["status"] == "network_error"
        assert summary.network_error == 1

    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript)
    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_no_content)
    def test_no_content_status(self, _art, _yt, tmp_db):
        """An article URL returning no content must get ``status='no_content'``."""
        urls = ["https://example.com/empty-page"]
        ingest_urls(urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert rows[0]["status"] == "no_content"

    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript)
    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    def test_summary_json_created(self, _art, _yt, tmp_db, sample_urls):
        """``ingest_summary.json`` must be written next to the database."""
        ingest_urls(sample_urls, db_path=tmp_db)

        summary_path = os.path.join(os.path.dirname(tmp_db), "ingest_summary.json")
        assert os.path.isfile(summary_path)

        with open(summary_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["total"] == 3
        assert "transcript_disabled" in data
        assert "video_unavailable" in data
        assert "network_error" in data

    @patch("src.ingest.youtube_fetcher.fetch_transcript", side_effect=_mock_fetch_transcript)
    @patch("src.ingest.trafilatura_fetcher.fetch_article", side_effect=_mock_fetch_article)
    def test_ingest_integration_youtube(self, _art, _yt, tmp_db):
        """Ingest a single YouTube URL and assert DB row status and raw_text."""
        urls = ["https://www.youtube.com/watch?v=kqtD5dpn9C8"]
        summary = ingest_urls(urls, db_path=tmp_db)

        conn = db.get_connection(tmp_db)
        rows = db.get_all_resources(conn)
        conn.close()

        assert len(rows) == 1
        assert rows[0]["status"] == "ok"
        assert rows[0]["raw_text"] is not None
        assert len(rows[0]["raw_text"]) > 0
        assert rows[0]["content_type"] == "youtube"
        assert summary.succeeded == 1


# ---------------------------------------------------------------------------
# Tests: Database Schema
# ---------------------------------------------------------------------------


class TestDatabaseSchema:
    """Verify the schema matches the spec."""

    def test_wal_mode(self, tmp_db):
        """WAL journal mode must be enabled."""
        db.migrate_db(tmp_db)
        conn = db.get_connection(tmp_db)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"

    def test_table_columns(self, tmp_db):
        """``RawResources`` must have exactly the columns from the spec."""
        db.migrate_db(tmp_db)
        conn = db.get_connection(tmp_db)
        cursor = conn.execute("PRAGMA table_info(RawResources);")
        columns = {row["name"] for row in cursor.fetchall()}
        conn.close()

        expected = {"id", "url", "content_type", "title", "raw_text",
                    "status", "extracted_at", "notes"}
        assert columns == expected

    def test_url_unique_constraint(self, tmp_db):
        """Inserting the same URL twice must not raise and must not duplicate."""
        db.migrate_db(tmp_db)
        conn = db.get_connection(tmp_db)

        id1 = db.insert_resource(conn, "https://x.com/a", "article", "T", "body", "ok")
        id2 = db.insert_resource(conn, "https://x.com/a", "article", "T", "body", "ok")

        rows = db.get_all_resources(conn)
        conn.close()

        assert len(rows) == 1
        assert id1 == id2


# ---------------------------------------------------------------------------
# Integration test (requires network — skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLiveYouTubeFetch:
    """Live integration test — requires internet access.

    Run with::

        pytest tests/test_ingest.py -v -m integration

    Skipped by default in CI (no network).
    """

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipped in CI — requires internet access",
    )
    def test_fetch_youtube_transcript_live(self):
        """Fetch transcript for kqtD5dpn9C8 over the network."""
        text, title, status = fetch_transcript(
            "https://www.youtube.com/watch?v=kqtD5dpn9C8"
        )
        assert status == "ok", f"Expected 'ok' but got '{status}'"
        assert text is not None
        assert len(text) > 100, "Transcript seems too short"
        assert title == "YouTube Video: kqtD5dpn9C8"
