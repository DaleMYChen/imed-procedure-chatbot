"""
Unit tests for src/llm.py — the ask() orchestration function.

All external I/O (Gemini API, Retriever) is mocked so tests run instantly
with no network access or model loading.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from src.llm import ask, BotResponse
from src.retriever import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(title: str, heading: str, url: str = None, score: float = 0.8) -> Chunk:
    url = url or f"https://i-med.com.au/procedures/{title.lower().replace(' ', '-')}"
    return Chunk(
        procedure_title=title,
        url=url,
        heading=heading,
        text=f"{heading}\nSome relevant content.",
        score=score,
    )


def _mock_gemini_response(text: str) -> MagicMock:
    """Build a mock that mimics google.generativeai model.generate_content() return."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestAskSuccess:

    def test_returns_bot_response_with_answer(self, sample_chunks, monkeypatch):
        """ask() returns a BotResponse with a non-empty answer on happy path."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = _mock_gemini_response(
            "Remove metal objects before the scan."
        )

        with (
            patch("src.llm.get_retriever", return_value=mock_retriever),
            patch("src.llm.genai.GenerativeModel", return_value=mock_model_instance),
        ):
            result = ask("How do I prepare for a CT scan?")

        assert isinstance(result, BotResponse)
        assert result.answer == "Remove metal objects before the scan."
        assert result.error is None

    def test_sources_deduplicated_by_url(self, monkeypatch):
        """Two chunks from the same URL produce only one source entry."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        shared_url = "https://i-med.com.au/procedures/ct-scan"
        chunks = [
            _make_chunk("CT Scan", "Overview",     url=shared_url),
            _make_chunk("CT Scan", "Preparation",  url=shared_url),
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = chunks

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = _mock_gemini_response("Answer.")

        with (
            patch("src.llm.get_retriever", return_value=mock_retriever),
            patch("src.llm.genai.GenerativeModel", return_value=mock_model_instance),
        ):
            result = ask("Tell me about CT scans")

        assert len(result.sources) == 1
        assert result.sources[0]["url"] == shared_url

    def test_sources_preserve_chunk_rank_order(self, monkeypatch):
        """Sources are listed in descending relevance order (chunk rank order)."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        chunks = [
            _make_chunk("CT Scan",  "Overview", score=0.9),
            _make_chunk("PET Scan", "Overview", score=0.6),
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = chunks

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = _mock_gemini_response("Answer.")

        with (
            patch("src.llm.get_retriever", return_value=mock_retriever),
            patch("src.llm.genai.GenerativeModel", return_value=mock_model_instance),
        ):
            result = ask("question")

        assert result.sources[0]["title"] == "CT Scan"
        assert result.sources[1]["title"] == "PET Scan"

    def test_answer_is_stripped_of_whitespace(self, sample_chunks, monkeypatch):
        """Trailing/leading whitespace in the model's response is stripped."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = _mock_gemini_response(
            "   Answer with spaces.   "
        )

        with (
            patch("src.llm.get_retriever", return_value=mock_retriever),
            patch("src.llm.genai.GenerativeModel", return_value=mock_model_instance),
        ):
            result = ask("question")

        assert result.answer == "Answer with spaces."


# ---------------------------------------------------------------------------
# Error condition A — no relevant chunks
# ---------------------------------------------------------------------------

class TestAskNoChunks:

    def test_returns_no_relevant_content_error(self, monkeypatch):
        """ask() returns error='no_relevant_content' when retriever finds nothing."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []   # nothing found

        with patch("src.llm.get_retriever", return_value=mock_retriever):
            result = ask("How do I fix a bike chain?")

        assert result.error == "no_relevant_content"
        assert result.sources == []
        assert "Angiography" in result.answer   # lists suggested topics

    def test_gemini_is_not_called_when_no_chunks(self, monkeypatch):
        """The Gemini API should NOT be invoked when retrieval returns nothing."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with (
            patch("src.llm.get_retriever", return_value=mock_retriever),
            patch("src.llm.genai.GenerativeModel") as mock_genai,
        ):
            ask("How do I fix a bike chain?")

        mock_genai.assert_not_called()


# ---------------------------------------------------------------------------
# Error condition B — missing API key
# ---------------------------------------------------------------------------

class TestAskMissingApiKey:

    def test_returns_api_key_missing_error(self, sample_chunks, monkeypatch):
        """ask() returns a clear error when GEMINI_API_KEY is not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        with patch("src.llm.get_retriever", return_value=mock_retriever):
            result = ask("How do I prepare for a CT scan?")

        assert result.error == "gemini_api_key_missing"
        assert result.sources == []


# ---------------------------------------------------------------------------
# Error condition C — Gemini API failures
# ---------------------------------------------------------------------------

class TestAskGeminiErrors:

    def test_google_api_error_returns_graceful_message(self, sample_chunks, monkeypatch):
        """GoogleAPIError is caught and returned as a BotResponse, not raised."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        from google.api_core.exceptions import GoogleAPIError

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = GoogleAPIError("quota exceeded")

        with (
            patch("src.llm.get_retriever", return_value=mock_retriever),
            patch("src.llm.genai.GenerativeModel", return_value=mock_model_instance),
        ):
            result = ask("question")

        assert "gemini_api_error" in result.error
        assert result.sources == []

    def test_unexpected_exception_returns_graceful_message(self, sample_chunks, monkeypatch):
        """Any unexpected exception from Gemini is caught and surfaced gracefully."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_chunks

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = ConnectionRefusedError("refused")

        with (
            patch("src.llm.get_retriever", return_value=mock_retriever),
            patch("src.llm.genai.GenerativeModel", return_value=mock_model_instance),
        ):
            result = ask("question")

        assert "gemini_unavailable" in result.error
        assert result.sources == []
        assert result.answer != ""   # user-friendly message, not a traceback