"""
Shared pytest fixtures used across unit and integration tests.
"""

import sys
from unittest.mock import MagicMock

# Must be before any src.* imports — prevents torch/sentence_transformers
# from initialising Apple MPS and triggering Abort trap: 6 on Mac
sys.modules['torch'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

import pytest

from src.retriever import Chunk


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