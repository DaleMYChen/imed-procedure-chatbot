"""
Unit tests for src/retriever.py

Each test uses pytest's tmp_path fixture for a unique PersistentClient database,
giving complete isolation without shared in-memory state between tests.

The procedures JSON is written to a real temp file so builtins.open is not
patched globally — chromadb's own file I/O is unaffected.

genai.embed_content is patched to return deterministic vectors so tests run
fully offline with no Gemini API key.
"""

import json
import math
import pytest
import numpy as np
import chromadb
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.retriever import Retriever, Chunk, get_retriever, MIN_SCORE, TOP_K, COLLECTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 768
_UNIT = 1.0 / math.sqrt(_DIM)   # uniform unit vector component


def _uniform_embed(model, content, task_type, **kwargs):
    """
    Fake embed_content: every input gets the same L2-normalised uniform vector.
    All stored chunks therefore have cosine similarity ≈ 1 with each other.
    **kwargs absorbs output_dimensionality and any future API parameters.
    """
    if isinstance(content, list):
        return {"embedding": [[_UNIT] * _DIM for _ in content]}
    return {"embedding": [_UNIT] * _DIM}


def _make_retriever(
    procedures_json: list,
    tmp_path: Path,
    embed_fn=None,
) -> Retriever:
    """
    Build a Retriever backed by a real PersistentClient at tmp_path/chroma.

    - procedures_json is written to tmp_path/procedures.json so builtins.open
      is NOT patched globally — chromadb's internal file I/O is unaffected.
    - DATA_PATH is patched to the real temp file (exists() returns True naturally).
    - CHROMA_PATH is patched to tmp_path/chroma for per-test DB isolation.
    - genai.embed_content is replaced by embed_fn (default: _uniform_embed).
    """
    data_file = tmp_path / "procedures.json"
    data_file.write_text(json.dumps(procedures_json), encoding="utf-8")
    chroma_dir = tmp_path / "chroma"

    with (
        patch("src.retriever.DATA_PATH", data_file),
        patch("src.retriever.CHROMA_PATH", chroma_dir),
        patch("src.retriever.genai.embed_content",
              side_effect=embed_fn or _uniform_embed),
    ):
        retriever = Retriever()
    return retriever


# ---------------------------------------------------------------------------
# Path B — _build_and_persist (collection empty on first run)
# ---------------------------------------------------------------------------

class TestBuildAndPersist:

    def test_documents_added_to_collection(self, sample_procedures_json, tmp_path):
        """All non-empty sections are stored as documents in the collection.

        sample_procedures_json: 2 CT Scan sections + 1 PET Scan section.
        Each body is short enough that the splitter produces one sub-chunk each.
        """
        retriever = _make_retriever(sample_procedures_json, tmp_path)
        assert retriever._collection.count() == 3

    def test_metadata_stored_correctly(self, sample_procedures_json, tmp_path):
        """procedure_title, url, and heading are present in every stored metadata dict."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)
        result = retriever._collection.get(include=["metadatas"])

        for meta in result["metadatas"]:
            assert "procedure_title" in meta
            assert "url" in meta
            assert "heading" in meta

        titles = {m["procedure_title"] for m in result["metadatas"]}
        assert "CT Scan" in titles
        assert "PET Scan" in titles

    def test_document_text_format(self, sample_procedures_json, tmp_path):
        """Each stored document is '<heading>\\n<body>' (heading prepended)."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)
        result = retriever._collection.get(include=["documents", "metadatas"])

        ct_overview_docs = [
            doc
            for doc, meta in zip(result["documents"], result["metadatas"])
            if meta["procedure_title"] == "CT Scan" and meta["heading"] == "Overview"
        ]
        assert len(ct_overview_docs) == 1
        assert ct_overview_docs[0] == "Overview\nA CT scan uses X-rays."

    def test_skips_procedures_with_error_flag(self, tmp_path):
        """Procedures with an 'error' field are not added to the collection."""
        procedures = [
            {
                "title": "Broken Page",
                "url": "https://example.com",
                "error": "Failed to retrieve page content.",
                "sections": {},
            }
        ]
        retriever = _make_retriever(procedures, tmp_path)
        assert retriever._collection.count() == 0

    def test_skips_empty_sections(self, tmp_path):
        """Sections with whitespace-only body produce no document in the collection."""
        procedures = [
            {
                "title": "CT Scan",
                "url": "https://example.com",
                "sections": {
                    "Overview": "Some text here.",
                    "Empty Section": "   ",
                },
            }
        ]
        retriever = _make_retriever(procedures, tmp_path)
        assert retriever._collection.count() == 1

        result = retriever._collection.get(include=["metadatas"])
        assert result["metadatas"][0]["heading"] == "Overview"

    def test_raises_if_data_file_missing(self, tmp_path):
        """FileNotFoundError is raised when procedures.json does not exist."""
        missing_file = tmp_path / "missing.json"   # not created → exists() = False
        chroma_dir = tmp_path / "chroma"

        with (
            patch("src.retriever.DATA_PATH", missing_file),
            patch("src.retriever.CHROMA_PATH", chroma_dir),
        ):
            with pytest.raises(FileNotFoundError, match="Data file not found"):
                Retriever()

    def test_no_chunks_held_in_memory_after_build(self, sample_procedures_json, tmp_path):
        """The Retriever has no _chunks attribute — everything lives in the collection."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)
        assert not hasattr(retriever, "_chunks")


# ---------------------------------------------------------------------------
# Path A — _load_from_store (collection already populated)
# ---------------------------------------------------------------------------

class TestLoadFromStore:

    def test_skips_embedding_when_collection_populated(self, tmp_path):
        """Path A: genai.embed_content is never called when collection already has data."""
        chroma_dir = tmp_path / "chroma"

        # Phase 1: pre-populate via a direct PersistentClient
        setup_client = chromadb.PersistentClient(path=str(chroma_dir))
        col = setup_client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["0"],
            documents=["Overview\nA CT scan uses X-rays."],
            embeddings=[[_UNIT] * _DIM],
            metadatas=[{
                "procedure_title": "CT Scan",
                "url": "https://i-med.com.au/procedures/ct-scan",
                "heading": "Overview",
            }],
        )
        # Release the connection before the Retriever opens its own
        del col, setup_client

        # Phase 2: Retriever detects existing data → path A → no embedding
        embed_mock = MagicMock()
        with (
            patch("src.retriever.DATA_PATH"),
            patch("src.retriever.CHROMA_PATH", chroma_dir),
            patch("src.retriever.genai.embed_content") as embed_mock,
        ):
            retriever = Retriever()

        embed_mock.assert_not_called()
        assert retriever._collection.count() == 1


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestRetrieve:

    def test_returns_chunks_above_min_score(self, sample_procedures_json, tmp_path):
        """Chunks with cosine similarity >= MIN_SCORE are returned."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)

        # Query aligned with stored uniform vectors → similarity ≈ 1.0
        with patch("src.retriever.genai.embed_content") as mock_embed:
            mock_embed.return_value = {"embedding": [_UNIT] * _DIM}
            results = retriever.retrieve("test query")

        assert len(results) > 0
        for chunk in results:
            assert chunk.score >= MIN_SCORE

    def test_empty_query_returns_empty_list(self, sample_procedures_json, tmp_path):
        """Empty or whitespace-only query short-circuits before any embedding."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)

        with patch("src.retriever.genai.embed_content") as mock_embed:
            assert retriever.retrieve("") == []
            assert retriever.retrieve("   ") == []
            mock_embed.assert_not_called()

    def test_respects_top_k(self, sample_procedures_json, tmp_path):
        """No more than top_k chunks are returned regardless of how many match."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)

        with patch("src.retriever.genai.embed_content") as mock_embed:
            mock_embed.return_value = {"embedding": [_UNIT] * _DIM}
            results = retriever.retrieve("test", top_k=1)

        assert len(results) <= 1

    def test_returned_chunks_have_correct_fields(self, sample_procedures_json, tmp_path):
        """Each returned Chunk has all required fields populated with the right types."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)

        with patch("src.retriever.genai.embed_content") as mock_embed:
            mock_embed.return_value = {"embedding": [_UNIT] * _DIM}
            results = retriever.retrieve("CT scan preparation")

        assert len(results) > 0
        for chunk in results:
            assert isinstance(chunk.procedure_title, str) and chunk.procedure_title
            assert isinstance(chunk.url, str) and chunk.url
            assert isinstance(chunk.heading, str) and chunk.heading
            assert isinstance(chunk.text, str) and chunk.text
            assert isinstance(chunk.score, float)

    def test_below_min_score_chunks_excluded(self, sample_procedures_json, tmp_path):
        """Chunks whose cosine similarity falls below MIN_SCORE are not returned.

        All stored embeddings are the uniform vector [unit, unit, ..., unit].
        A query vector of [1, 0, 0, ..., 0] has cosine similarity = 1/sqrt(768)
        ≈ 0.036 with each stored chunk — below MIN_SCORE = 0.25.
        """
        retriever = _make_retriever(sample_procedures_json, tmp_path)

        low_sim_query = [1.0] + [0.0] * (_DIM - 1)

        with patch("src.retriever.genai.embed_content") as mock_embed:
            mock_embed.return_value = {"embedding": low_sim_query}
            results = retriever.retrieve("completely unrelated query")

        assert results == []

    def test_scores_attached_to_returned_chunks(self, sample_procedures_json, tmp_path):
        """score on returned Chunk equals 1 - cosine_distance; finite and above threshold."""
        retriever = _make_retriever(sample_procedures_json, tmp_path)

        with patch("src.retriever.genai.embed_content") as mock_embed:
            mock_embed.return_value = {"embedding": [_UNIT] * _DIM}
            results = retriever.retrieve("any question")

        for chunk in results:
            assert isinstance(chunk.score, float)
            assert chunk.score >= MIN_SCORE
            # Allow slight fp excess above 1.0 from HNSW cosine distance rounding
            assert chunk.score < 1.01


# ---------------------------------------------------------------------------
# Singleton — get_retriever()
# ---------------------------------------------------------------------------

class TestGetRetriever:

    def test_returns_same_instance_on_repeated_calls(
        self, sample_procedures_json, tmp_path
    ):
        """get_retriever() must return the identical object on every call."""
        import src.retriever as retriever_module

        data_file = tmp_path / "procedures.json"
        data_file.write_text(json.dumps(sample_procedures_json), encoding="utf-8")
        chroma_dir = tmp_path / "chroma"

        original_instance = retriever_module._retriever_instance
        try:
            with (
                patch("src.retriever.DATA_PATH", data_file),
                patch("src.retriever.CHROMA_PATH", chroma_dir),
                patch("src.retriever.genai.embed_content",
                      side_effect=_uniform_embed),
            ):
                retriever_module._retriever_instance = None
                r1 = get_retriever()
                r2 = get_retriever()

            assert r1 is r2
        finally:
            retriever_module._retriever_instance = original_instance
