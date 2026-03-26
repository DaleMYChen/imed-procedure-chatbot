"""
Integration tests for src/main.py FastAPI endpoints.

The FastAPI app is exercised via httpx's AsyncClient / TestClient — a real
HTTP request/response cycle runs through routing, validation, and serialisation.
Only ask() (the LLM+retriever boundary) is mocked, keeping the test focused on
the API layer without needing Gemini or SentenceTransformer.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import the app — retriever singleton must not be initialised at import time
# (it's created lazily inside ask(), which we mock below).
from src.main import app
from src.llm import BotResponse


# ---------------------------------------------------------------------------
# Shared client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """FastAPI TestClient — reused across all tests in this module."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mock_ask(answer, sources=None, error=None):
    if sources is None:
        sources = [{"title": "CT Scan", "url": "https://i-med.com.au/procedures/ct-scan", "section": "Overview"}]
    bot_response = BotResponse(answer=answer, sources=sources, error=error)
    return patch("src.main.ask", return_value=bot_response)

# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRootEndpoint:

    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_html_content_type(self, client):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_html_body_contains_title(self, client):
        response = client.get("/")
        assert "I-MED" in response.text


# ---------------------------------------------------------------------------
# POST /api/ask — happy path
# ---------------------------------------------------------------------------

class TestAskEndpointSuccess:

    def test_returns_200_with_valid_payload(self, client):
        with _mock_ask("Fast for 4 hours if contrast is used."):
            response = client.post("/api/ask", json={"question": "How do I prepare for a CT scan?"})

        assert response.status_code == 200

    def test_response_matches_answer_response_schema(self, client):
        """Response body must conform to AnswerResponse: answer, sources, error."""
        with _mock_ask("Fast for 4 hours if contrast is used."):
            response = client.post("/api/ask", json={"question": "CT prep?"})

        body = response.json()
        assert "answer" in body
        assert "sources" in body
        assert "error" in body

    def test_answer_text_returned_correctly(self, client):
        expected = "You must fast for 4 hours."
        with _mock_ask(expected):
            response = client.post("/api/ask", json={"question": "Any fasting required?"})

        assert response.json()["answer"] == expected

    def test_sources_list_returned_correctly(self, client):
        sources = [
            {"title": "CT Scan", "url": "https://i-med.com.au/procedures/ct-scan", "section": "Preparation"}
        ]
        with _mock_ask("Some answer.", sources=sources):
            response = client.post("/api/ask", json={"question": "CT prep?"})

        body_sources = response.json()["sources"]
        assert len(body_sources) == 1
        assert body_sources[0]["title"] == "CT Scan"
        assert body_sources[0]["section"] == "Preparation"

    def test_error_is_none_on_success(self, client):
        with _mock_ask("Some answer.", error=None):
            response = client.post("/api/ask", json={"question": "CT prep?"})

        assert response.json()["error"] is None

    def test_no_relevant_content_still_returns_200(self, client):
        """Out-of-scope questions return 200 (not an HTTP error), with error field set."""
        with _mock_ask(
            "I couldn't find relevant information.",
            sources=[],
            error="no_relevant_content",
        ):
            response = client.post("/api/ask", json={"question": "Fix my bike?"})

        assert response.status_code == 200
        body = response.json()
        assert body["error"] == "no_relevant_content"
        assert body["sources"] == []


# ---------------------------------------------------------------------------
# POST /api/ask — Gemini errors → 503
# ---------------------------------------------------------------------------

class TestAskEndpointGeminiErrors:

    def test_gemini_api_error_returns_503(self, client):
        """When ask() returns a gemini_* error, the endpoint raises HTTP 503."""
        bot_response = BotResponse(
            answer="Could not reach the language model.",
            sources=[],
            error="gemini_api_error: quota exceeded",
        )
        with patch("src.main.ask", return_value=bot_response):
            response = client.post("/api/ask", json={"question": "CT prep?"})

        assert response.status_code == 503

    def test_gemini_unavailable_returns_503(self, client):
        bot_response = BotResponse(
            answer="Could not reach the language model.",
            sources=[],
            error="gemini_unavailable: connection refused",
        )
        with patch("src.main.ask", return_value=bot_response):
            response = client.post("/api/ask", json={"question": "CT prep?"})

        assert response.status_code == 503

    def test_503_detail_contains_answer_message(self, client):
        """The 503 response body's detail field should surface the user-friendly message."""
        user_message = "Could not reach the language model. Please try again."
        bot_response = BotResponse(
            answer=user_message,
            sources=[],
            error="gemini_api_error: bad gateway",
        )
        with patch("src.main.ask", return_value=bot_response):
            response = client.post("/api/ask", json={"question": "CT prep?"})

        assert response.json()["detail"] == user_message


# ---------------------------------------------------------------------------
# POST /api/ask — request validation
# ---------------------------------------------------------------------------

class TestAskEndpointValidation:

    def test_missing_question_field_returns_422(self, client):
        """Pydantic validation: missing required 'question' field → 422."""
        response = client.post("/api/ask", json={})
        assert response.status_code == 422

    def test_wrong_content_type_returns_422(self, client):
        """Sending plain text instead of JSON → 422."""
        response = client.post(
            "/api/ask",
            content="How do I prepare?",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 422

    def test_empty_json_body_returns_422(self, client):
        response = client.post("/api/ask", json=None)
        assert response.status_code == 422

    def test_ask_is_called_with_correct_question(self, client):
        """The question string from the request is passed verbatim to ask()."""
        question = "Am I eligible for lung screening?"
        with _mock_ask("Some answer.") as mock_ask_fn:
            client.post("/api/ask", json={"question": question})

        mock_ask_fn.assert_called_once_with(question)