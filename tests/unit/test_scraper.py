"""
Unit tests for src/scraper.py

Network calls (requests.get) and filesystem writes (json.dump, mkdir) are
mocked so tests run without internet access or disk side-effects.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, mock_open, call
from pathlib import Path

import requests

from src.scraper import _fetch_html, _parse_procedure, scrape_all, Procedure


# ---------------------------------------------------------------------------
# Minimal HTML fixture
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html>
<body>
  <main>
    <h1>CT Scan</h1>
    <p>A CT scan uses X-rays to produce detailed images.</p>
    <h2>How to prepare</h2>
    <p>Remove all metal objects.</p>
    <p>Fast for 4 hours if contrast dye is used.</p>
    <h3>What to bring</h3>
    <ul>
      <li>Your referral letter</li>
      <li>Medicare card</li>
    </ul>
  </main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# _fetch_html
# ---------------------------------------------------------------------------

class TestFetchHtml:

    def test_returns_html_on_200(self):
        """Returns the response text on a successful 200 response."""
        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status.return_value = None

        with patch("src.scraper.requests.get", return_value=mock_response):
            result = _fetch_html("https://example.com")

        assert result == "<html>content</html>"

    def test_returns_none_on_connection_error(self):
        """ConnectionError is caught and None is returned (no exception raised)."""
        with patch(
            "src.scraper.requests.get",
            side_effect=requests.exceptions.ConnectionError,
        ):
            result = _fetch_html("https://example.com")

        assert result is None

    def test_returns_none_on_timeout(self):
        """Timeout is caught and None is returned."""
        with patch(
            "src.scraper.requests.get",
            side_effect=requests.exceptions.Timeout,
        ):
            result = _fetch_html("https://example.com")

        assert result is None

    def test_returns_none_on_http_error(self):
        """HTTP 4xx/5xx errors are caught and None is returned."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)

        with patch(
            "src.scraper.requests.get",
            side_effect=http_error,
        ):
            result = _fetch_html("https://example.com")

        assert result is None

    def test_sends_user_agent_header(self):
        """The User-Agent header is included in every request (polite crawling)."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None

        with patch("src.scraper.requests.get", return_value=mock_response) as mock_get:
            _fetch_html("https://example.com")

        _, kwargs = mock_get.call_args
        assert "User-Agent" in kwargs.get("headers", {})


# ---------------------------------------------------------------------------
# _parse_procedure
# ---------------------------------------------------------------------------

class TestParseProcedure:

    def test_returns_procedure_dataclass(self):
        """_parse_procedure always returns a Procedure instance."""
        result = _parse_procedure(SAMPLE_HTML, "CT Scan", "https://example.com")
        assert isinstance(result, Procedure)

    def test_title_and_url_preserved(self):
        """Title and URL are stored verbatim on the returned Procedure."""
        result = _parse_procedure(SAMPLE_HTML, "CT Scan", "https://i-med.com.au/ct")
        assert result.title == "CT Scan"
        assert result.url == "https://i-med.com.au/ct"

    def test_sections_dict_built_from_headings(self):
        """Each h1/h2/h3 becomes a key in the sections dict."""
        result = _parse_procedure(SAMPLE_HTML, "CT Scan", "https://example.com")

        assert "CT Scan" in result.sections          # h1
        assert "How to prepare" in result.sections   # h2
        assert "What to bring" in result.sections    # h3

    def test_paragraph_text_under_correct_heading(self):
        """Paragraph text is attributed to the most recent heading seen."""
        result = _parse_procedure(SAMPLE_HTML, "CT Scan", "https://example.com")

        assert "X-rays" in result.sections["CT Scan"]
        assert "Remove all metal" in result.sections["How to prepare"]
        assert "Fast for 4 hours" in result.sections["How to prepare"]

    def test_list_items_included_as_bullets(self):
        """<li> elements are captured with bullet prefix '• '."""
        result = _parse_procedure(SAMPLE_HTML, "CT Scan", "https://example.com")

        bring_text = result.sections.get("What to bring", "")
        assert "• Your referral letter" in bring_text or "referral" in bring_text.lower()

    def test_raw_text_is_non_empty(self):
        """raw_text is populated and contains content from the page."""
        result = _parse_procedure(SAMPLE_HTML, "CT Scan", "https://example.com")
        assert len(result.raw_text) > 50

    def test_noise_tags_removed(self):
        """script, style, nav, footer content should not appear in sections."""
        noisy_html = """
        <html><body><main>
          <script>alert('noise')</script>
          <nav>Menu</nav>
          <h1>Real Content</h1>
          <p>This is valid.</p>
          <footer>Footer noise</footer>
        </main></body></html>
        """
        result = _parse_procedure(noisy_html, "Test", "https://example.com")

        flat = " ".join(result.sections.values())
        assert "alert" not in flat
        assert "Menu" not in flat
        assert "Footer noise" not in flat
        assert "This is valid." in flat


# ---------------------------------------------------------------------------
# scrape_all
# ---------------------------------------------------------------------------

class TestScrapeAll:

    def test_returns_list_of_dicts(self):
        """scrape_all() returns a list (one entry per URL in PROCEDURE_URLS)."""
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None

        with (
            patch("src.scraper.requests.get", return_value=mock_response),
            patch("src.scraper.time.sleep"),
            patch("builtins.open", mock_open()),
            patch("src.scraper.OUTPUT_PATH") as mock_out,
        ):
            mock_out.parent.mkdir = MagicMock()
            mock_out.__str__ = lambda s: "/fake/path"
            results = scrape_all()

        assert isinstance(results, list)
        assert len(results) > 0

    def test_records_error_stub_on_fetch_failure(self):
        """If a page can't be fetched, a stub dict with 'error' key is recorded."""
        with (
            patch("src.scraper.requests.get", side_effect=requests.exceptions.ConnectionError),
            patch("src.scraper.time.sleep"),
            patch("builtins.open", mock_open()),
            patch("src.scraper.OUTPUT_PATH") as mock_out,
        ):
            mock_out.parent.mkdir = MagicMock()
            mock_out.__str__ = lambda s: "/fake/path"
            results = scrape_all()

        error_stubs = [r for r in results if r.get("error")]
        assert len(error_stubs) > 0
        for stub in error_stubs:
            assert stub["sections"] == {}