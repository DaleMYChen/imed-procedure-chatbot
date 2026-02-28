# 1. loads json & chunk by sections
# 2. vectorise with sentence-transformer embedding model
# 3. implements semantic search on retrieval

"""
Inspired by the memory_builder pattern in:
  https://github.com/umbertogriffo/rag-chatbot
but simplified: no vector DB dependency — embeddings are held in memory
(fine for 6 pages / ~50 chunks).
"""

from __future__ import annotations
from typing import List, Optional  # type hints

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "procedure_data" / "procedures.json"
EMBED_MODEL = "all-MiniLM-L6-v2"   # ~80 MB, fast CPU inference, 384-dim vectors
TOP_K = 7                           # #most similar chunks
MIN_SCORE = 0.25                    # cosine similarity threshold (0–1)
                                    # below this → "no relevant content" error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk object formatting
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """One retrievable unit: a (heading, body) section from a procedure page."""
    procedure_title: str
    url: str
    heading: str
    text: str                        # heading + body, used for embedding
    score: float = 0.0               # populated after retrieval


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """
    Loads procedures.json, splits each page into section-level chunks,
    embeds them once at startup, and supports semantic search at query time.
    """

    def __init__(self):
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        self._model = SentenceTransformer(EMBED_MODEL)  # class-internal vars
        self._chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._load_and_embed()

    ### Load json, chunk, embed
    def _load_and_embed(self) -> None:
        """Load procedures.json → build chunks → embed all at once.
        Small json data: no vector database required.
        """

        # --- ensure pages are available --------------------------------
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Data file not found at {DATA_PATH}. "
                "Please run `python src/scraper.py` first."
            )

        with open(DATA_PATH, encoding="utf-8") as f:
            procedures: List[dict] = json.load(f)

        for proc in procedures:
            # Skip pages that failed during scraping
            if proc.get("error") or not proc.get("sections"):
                logger.warning("Skipping %s (no content).", proc.get("title"))
                continue

            for heading, body in proc["sections"].items():
                if not body.strip():  # skip empty sections
                    continue
                # Chunk text = heading + body so the embedding carries topic context
                chunk_text = f"{heading}\n{body}"
                self._chunks.append(
                    Chunk(
                        procedure_title=proc["title"],
                        url=proc["url"],
                        heading=heading,
                        text=chunk_text,
                    )
                )


        logger.info("Embedding %d chunks...", len(self._chunks))
        texts = [c.text for c in self._chunks]
        # encode() returns shape (n_chunks, 384)
        self._embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # pre-normalise → cosine sim = dot product
        )
        logger.info("Retriever ready.")


    ### Semantic search & retrieve
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        """
        Embed the query and return up to `top_k` chunks ranked by cosine similarity.
        """

        # Case: no relevant content found
        if not query.strip():
            return []

        # Embed query
        query_vec = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  

        # Cosine similarity
        scores: np.ndarray = self._embeddings @ query_vec  # shape: (n_chunks,)

        # Get top_k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[Chunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < MIN_SCORE:
                break   # remaining chunks will have lower similarities
            chunk = self._chunks[idx]
            chunk.score = score
            results.append(chunk)

        if not results:
            logger.info("No chunks exceeded MIN_SCORE=%.2f for query: %r", MIN_SCORE, query)

        return results


# ---------------------------------------------------------------------------
# Module-level singleton (imported by llm.py)
# ---------------------------------------------------------------------------

# module-level variable: embed once, used for every user request
_retriever_instance: Optional[Retriever] = None

def get_retriever() -> Retriever:
    """Lazy singleton — model loads once, reused across all requests."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance