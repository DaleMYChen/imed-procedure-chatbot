"""
Unit tests for src/retriever.py

All external dependencies (SentenceTransformer, filesystem) are mocked so
tests run fast with no model download or disk access required.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from src.retriever import Retriever, Chunk, get_retriever, MIN_SCORE, TOP_K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever(procedures_json: list) -> Retriever:
    """
    Instantiate a Retriever with fully mocked filesystem and embedding model.
    Returns a Retriever whose _embeddings are deterministic unit vectors.
    """
    # Deterministic fake embeddings: one 4-dim unit vector per chunk
    # (the real model uses 384-dim; dimensionality doesn't matter for tests)
    n_sections = sum(len(p["sections"]) for p in procedures_json if not p.get("error"))

    fake_embeddings = np.eye(n_sections, 4, dtype=np.float32)  # orthogonal unit vecs

    mock_model = MagicMock()
    # encode() is called once for all chunk texts, then once per query
    mock_model.encode.side_effect = [
        fake_embeddings,                   # bulk embed at startup
        # individual query embeds are set in each test
    ]

    with (
        patch("src.retriever.DATA_PATH") as mock_path,
        patch("src.retriever.SentenceTransformer", return_value=mock_model),
        patch("builtins.open", mock_open(read_data=json.dumps(procedures_json))),
    ):
        mock_path.exists.return_value = True
        retriever = Retriever()

    retriever._model = mock_model   # keep a reference for per-test encode() setup
    return retriever


# ---------------------------------------------------------------------------
# _load_and_embed
# ---------------------------------------------------------------------------

class TestLoadAndEmbed:

    def test_chunks_created_from_sections(self, sample_procedures_json):
        """One Chunk is created for each non-empty section across all procedures."""
        retriever = _make_retriever(sample_procedures_json)

        # CT Scan has 2 sections, PET Scan has 1 → 3 chunks total
        assert len(retriever._chunks) == 3

    def test_chunk_fields_populated(self, sample_procedures_json):
        """Chunk fields are correctly mapped from the JSON source."""
        retriever = _make_retriever(sample_procedures_json)

        ct_chunks = [c for c in retriever._chunks if c.procedure_title == "CT Scan"]
        assert len(ct_chunks) == 2

        headings = {c.heading for c in ct_chunks}
        assert "Overview" in headings
        assert "Preparation" in headings

    def test_chunk_text_combines_heading_and_body(self, sample_procedures_json):
        """chunk.text = '<heading>\\n<body>' for embedding context."""
        retriever = _make_retriever(sample_procedures_json)

        overview_chunk = next(
            c for c in retriever._chunks
            if c.heading == "Overview" and c.procedure_title == "CT Scan"
        )
        assert overview_chunk.text == "Overview\nA CT scan uses X-rays."

    def test_skips_procedures_with_error_flag(self):
        """Procedures with an 'error' field in JSON are skipped entirely."""
        procedures = [
            {
                "title": "Broken Page",
                "url": "https://example.com",
                "error": "Failed to retrieve page content.",
                "sections": {},
            }
        ]
        # Provide an empty embedding so the call doesn't fail
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((0, 4), dtype=np.float32)

        with (
            patch("src.retriever.DATA_PATH") as mock_path,
            patch("src.retriever.SentenceTransformer", return_value=mock_model),
            patch("builtins.open", mock_open(read_data=json.dumps(procedures))),
        ):
            mock_path.exists.return_value = True
            retriever = Retriever()

        assert retriever._chunks == []

    def test_skips_empty_sections(self):
        """Sections with blank body text are not turned into chunks."""
        procedures = [
            {
                "title": "CT Scan",
                "url": "https://example.com",
                "sections": {
                    "Overview": "Some text here.",
                    "Empty Section": "   ",   # whitespace only — should be skipped
                },
            }
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.eye(1, 4, dtype=np.float32)

        with (
            patch("src.retriever.DATA_PATH") as mock_path,
            patch("src.retriever.SentenceTransformer", return_value=mock_model),
            patch("builtins.open", mock_open(read_data=json.dumps(procedures))),
        ):
            mock_path.exists.return_value = True
            retriever = Retriever()

        assert len(retriever._chunks) == 1
        assert retriever._chunks[0].heading == "Overview"

    def test_raises_if_data_file_missing(self):
        """FileNotFoundError is raised when procedures.json does not exist."""
        with (
            patch("src.retriever.DATA_PATH") as mock_path,
            patch("src.retriever.SentenceTransformer"),
        ):
            mock_path.exists.return_value = False
            with pytest.raises(FileNotFoundError, match="Data file not found"):
                Retriever()

# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestRetrieve:

    def _retriever_with_query_vec(self, procedures_json, query_vec: np.ndarray):
        """
        Helper: build a Retriever then configure query embed to return query_vec.
        Because _make_retriever consumes one encode() call for bulk embed,
        we add the query vec as the next side_effect entry.
        """
        retriever = _make_retriever(procedures_json)
        # Append query result to existing side_effect list
        retriever._model.encode.side_effect = [query_vec]
        return retriever

    def test_returns_top_matching_chunk(self, sample_procedures_json):
        """The chunk with the highest cosine similarity is returned first."""
        retriever = _make_retriever(sample_procedures_json)
        # Query vector aligned with chunk index 0 (first eye row = [1,0,0,0])
        retriever._model.encode.side_effect = [np.array([1, 0, 0, 0], dtype=np.float32)]

        results = retriever.retrieve("test query")

        assert len(results) >= 1
        # First result should have the highest score
        assert results[0].score >= results[-1].score

    def test_scores_attached_to_returned_chunks(self, sample_procedures_json):
        """Each returned Chunk has score populated from cosine similarity."""
        retriever = _make_retriever(sample_procedures_json)
        retriever._model.encode.side_effect = [np.array([1, 0, 0, 0], dtype=np.float32)]

        results = retriever.retrieve("any question")

        for chunk in results:
            assert isinstance(chunk.score, float)
            assert chunk.score >= MIN_SCORE

    def test_empty_query_returns_empty_list(self, sample_procedures_json):
        """Empty or whitespace-only query short-circuits before embedding."""
        retriever = _make_retriever(sample_procedures_json)

        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_below_min_score_chunks_excluded(self, sample_procedures_json):
        """Chunks with cosine similarity < MIN_SCORE are not returned."""
        retriever = _make_retriever(sample_procedures_json)
        # Zero vector → all dot products = 0, which is below MIN_SCORE=0.25
        retriever._model.encode.side_effect = [np.zeros(4, dtype=np.float32)]

        results = retriever.retrieve("completely unrelated")

        assert results == []

    def test_respects_top_k(self, sample_procedures_json):
        """No more than top_k chunks are returned."""
        retriever = _make_retriever(sample_procedures_json)
        # High uniform scores: all chunks should be above threshold
        retriever._model.encode.side_effect = [
            np.ones(4, dtype=np.float32) / np.sqrt(4)
        ]

        results = retriever.retrieve("something", top_k=2)

        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestGetRetriever:

    def test_returns_same_instance_on_repeated_calls(self, sample_procedures_json):
        """get_retriever() must return the same object (lazy singleton)."""
        import src.retriever as retriever_module

        mock_model = MagicMock()
        mock_model.encode.return_value = np.eye(3, 4, dtype=np.float32)

        with (
            patch("src.retriever.DATA_PATH") as mock_path,
            patch("src.retriever.SentenceTransformer", return_value=mock_model),
            patch("builtins.open", mock_open(read_data=json.dumps(sample_procedures_json))),
        ):
            mock_path.exists.return_value = True
            # Reset the global singleton so we can test fresh creation
            retriever_module._retriever_instance = None

            r1 = get_retriever()
            r2 = get_retriever()

        assert r1 is r2
        # SentenceTransformer should have been instantiated only once
        assert mock_model.encode.call_count == 1