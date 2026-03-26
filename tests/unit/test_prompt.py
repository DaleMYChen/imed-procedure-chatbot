"""
Unit tests for the prompt builder in src/llm.py

_build_prompt is a pure function with no external dependencies,
so no mocking is required — tests run entirely in memory.
"""

import pytest
from src.retriever import Chunk
from src.llm import _build_prompt, MAX_CONTEXT_CHARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(title: str, heading: str, body: str, score: float = 0.9) -> Chunk:
    return Chunk(
        procedure_title=title,
        url=f"https://i-med.com.au/procedures/{title.lower().replace(' ', '-')}",
        heading=heading,
        text=f"{heading}\n{body}",
        score=score,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:

    def test_question_appears_in_prompt(self):
        """The user's question must be present verbatim in the built prompt."""
        chunk = _make_chunk("CT Scan", "Preparation", "Remove metal objects.")
        prompt = _build_prompt("How do I prepare?", [chunk])

        assert "How do I prepare?" in prompt

    def test_chunk_source_label_included(self):
        """Each chunk is labelled with its procedure title and section heading."""
        chunk = _make_chunk("CT Scan", "Preparation", "Remove metal objects.")
        prompt = _build_prompt("Prep?", [chunk])

        assert "[Source: CT Scan — Preparation]" in prompt

    def test_chunk_body_text_included(self):
        """The actual body text of each chunk appears in the prompt context."""
        chunk = _make_chunk("CT Scan", "Preparation", "Remove metal objects.")
        prompt = _build_prompt("Prep?", [chunk])

        assert "Remove metal objects." in prompt

    def test_multiple_chunks_all_included_within_limit(self):
        """Multiple short chunks are all included when total chars < MAX_CONTEXT_CHARS."""
        chunks = [
            _make_chunk("CT Scan", "Overview", "A CT uses X-rays."),
            _make_chunk("PET Scan", "Overview", "A PET measures metabolism."),
        ]
        prompt = _build_prompt("What scans exist?", chunks)

        assert "CT Scan" in prompt
        assert "PET Scan" in prompt

    def test_truncates_when_context_exceeds_max_chars(self):
        """Chunks that would push context past MAX_CONTEXT_CHARS are omitted."""
        # Chunk 1: just under half the limit
        big_body = "x" * (MAX_CONTEXT_CHARS // 2 - 50)
        chunk1 = _make_chunk("CT Scan", "Overview", big_body)

        # Chunk 2: would push total over the limit
        chunk2 = _make_chunk("PET Scan", "Overview", "x" * (MAX_CONTEXT_CHARS // 2 + 200))

        # Chunk 3: small, but comes after chunk2 — should also be excluded because
        # the loop breaks once char_count hits the ceiling
        chunk3 = _make_chunk("Angiography", "Overview", "Small text.")

        prompt = _build_prompt("question", [chunk1, chunk2, chunk3])

        assert "CT Scan" in prompt
        # chunk2 and chunk3 should not appear (budget was exhausted by chunk2)
        assert "PET Scan" not in prompt
        assert "Angiography" not in prompt

    def test_prompt_contains_system_instruction(self):
        """The grounding instruction ('Answer ONLY based on the context') is present."""
        chunk = _make_chunk("CT Scan", "Overview", "Uses X-rays.")
        prompt = _build_prompt("anything", [chunk])

        assert "Answer ONLY based on the context" in prompt

    def test_prompt_contains_fallback_instruction(self):
        """The fallback URL instruction for out-of-scope answers is present."""
        chunk = _make_chunk("CT Scan", "Overview", "Uses X-rays.")
        prompt = _build_prompt("anything", [chunk])

        assert "i-med.com.au/contact-us" in prompt

    def test_empty_chunks_list_produces_empty_context(self):
        """If no chunks are passed, the Context block in the prompt is empty."""
        prompt = _build_prompt("What is a CT scan?", [])

        # Context section should exist but have no source labels
        assert "Context:" in prompt
        assert "[Source:" not in prompt