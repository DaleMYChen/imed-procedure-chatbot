"""
Shared pytest fixtures used across unit and integration tests.

Extension 2 changes vs original:
  - Removed sys.modules mocks for torch and sentence_transformers (no longer
    a dependency after switching to Google embedding-004).
  - Added _mock_embed fixture that patches genai.embed_content to return fake
    numpy-compatible vectors, keeping all tests fully offline.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.retriever import Chunk


# ---------------------------------------------------------------------------
# Embedding mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embed():
    """
    Patches genai.embed_content to return deterministic fake vectors so tests
    run offline with no Gemini API key required.

    Returns a fake 768-dim embedding for both document and query calls:
      - list input  (index time) → {"embedding": [[0.1, 0.1, ...], ...]}
      - str input   (query time) → {"embedding": [0.1, 0.1, ...]}

    The vectors are uniform and L2-normalised so dot-product scores are
    consistent and above MIN_SCORE=0.25 for retriever tests.
    """
    def _fake_embed(model, content, task_type, **kwargs):
        dim = 768
        unit = 1.0 / (dim ** 0.5)          # produces a unit-norm vector
        if isinstance(content, list):
            return {"embedding": [[unit] * dim for _ in content]}
        return {"embedding": [unit] * dim}

    with patch("src.retriever.genai.embed_content", side_effect=_fake_embed) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunk():
    """A single Chunk with realistic field values."""
    return Chunk(
        procedure_title="CT Scan",
        url="https://i-med.com.au/procedures/ct-scan",
        heading="How do I prepare for a CT scan?",
        text="How do I prepare for a CT scan?\nRemove metal objects. Fast for 4 hours if contrast is used.",
        score=0.82,
    )


@pytest.fixture
def sample_chunks(sample_chunk):
    """A small list of Chunks from two different procedures."""
    chunk2 = Chunk(
        procedure_title="PET Scan",
        url="https://i-med.com.au/procedures/pet-scan",
        heading="How long does a PET scan take?",
        text="How long does a PET scan take?\nExpect 2–3 hours in the department.",
        score=0.61,
    )
    return [sample_chunk, chunk2]


@pytest.fixture
def sample_procedures_json():
    """Minimal procedures.json structure for retriever loading tests."""
    return [
        {
            "title": "CT Scan",
            "url": "https://i-med.com.au/procedures/ct-scan",
            "raw_text": "Overview\nA CT scan uses X-rays.",
            "sections": {
                "Overview": "A CT scan uses X-rays.",
                "Preparation": "Remove metal objects before the scan.",
            },
        },
        {
            "title": "PET Scan",
            "url": "https://i-med.com.au/procedures/pet-scan",
            "raw_text": "Overview\nA PET scan measures metabolic activity.",
            "sections": {
                "Overview": "A PET scan measures metabolic activity.",
            },
        },
    ]