# 1. loads json & chunk by sections using LangChain RecrusiveCharacterTextSplitter
# 2. vectorise with Google gemini-embedding-001 via Gemini API
# 3. implements semantic search on retrieval

"""
Original design: no vector DB dependency — embeddings are held in memory

Extension 1: Semantic chunking
  - Heading-body pairs are now sub-chunked with LangChain's
    RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50).
    This prevents long section bodies from overflowing the embedding model's
    token limit and produces semantically tighter retrieval units.

  - SentenceTransformer (all-MiniLM-L6-v2, ~80 MB, requires torch) is replaced
    by Google's gemini-embedding-001 via the Gemini API. This removes the ~2 GB
    torch dependency from the Docker image and keeps all Google API calls under
    a single key/provider.

  - Embedding at index time uses task_type="RETRIEVAL_DOCUMENT"; at query time
    task_type="RETRIEVAL_QUERY". output_dimensionality=768 is set explicitly:
    gemini-embedding-001 defaults to 3072 dims; 768 is the recommended compact
    size (0.26% quality loss vs full 3072). Vectors are L2-normalised by the
    API, so cosine similarity is still computed as a dot product.

    
Extension 2. ChromaDB persistent vector storage
    - self.embeddings (arr) replaced by a ChromaDB PersistentClient collection
      written to chroma_store/.

    - Two startup paths:
      A. collection already populated (same container restart): 
         Retrieve straightaway.

      B. collection empty (first run):
         full split-embed-persist pipeline. 

    - retrieve() conducts similarity search to Chroma's HNSW index. 
      sim score = 1 - distance. 
"""

from __future__ import annotations
from typing import List, Optional  # type hints

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "procedure_data" / "procedures.json"
CHROMA_PATH = Path(__file__).parent.parent / "chroma_store"  # persist to disk
COLLECTION = "procedures"

EMBED_MODEL = "models/gemini-embedding-001"  # default 3072-dim; we request 768
EMBED_DIM   = 768                           # output_dimensionality passed to API
TOP_K = 3                           # #most similar chunks
MIN_SCORE = 0.25                    # cosine similarity threshold (0–1)
                                    # below this → "no relevant content" error

# Splitter config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

logger = logging.getLogger(__name__)

# Configure Gemini client once at import time (shared with llm.py)
_api_key = os.environ.get("GEMINI_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)
else:
    logging.warning("GEMINI_API_KEY not set. Retriever initialisation will fail.")

# ---------------------------------------------------------------------------
# Chunk dataclass formatting
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """
    Original setup: One retrievable unit was a (heading, body) section from a page.
    Extended setup: One retrievable unit produced by splitting a procedure section.
    """
    procedure_title: str
    url: str
    heading: str                     # parent section heading — preserved for source citations
    text: str                        # heading + sub-chunk body, used for embedding
    score: float = 0.0               # populated after retrieval



# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _embed_documents(texts: List[str]) -> np.ndarray:
    """
    Embed a list of document strings with task_type RETRIEVAL_DOCUMENT.
    Returns a float32 array of shape (n, 768), one row per chunk (text).
 
    The Gemini embed_content call accepts a list directly; the SDK returns
    {"embedding": [[...], [...]]} when given a list.
    """
    result = genai.embed_content(
        model = EMBED_MODEL,
        content = texts,
        task_type = 'RETRIEVAL_DOCUMENT',
        output_dimensionality = EMBED_DIM,
    )
    return np.array(result["embedding"], dtype = np.float32)


def _embed_query(text:str) -> np.ndarray:
    """
    Embed a single query string with task_type RETRIEVAL_QUERY.
    Returns a float32 array of shape (768,).
    """
    result = genai.embed_content(
        model = EMBED_MODEL,
        content = text,
        task_type = "RETRIEVAL_QUERY",
        output_dimensionality = EMBED_DIM,
    )
    return np.array(result["embedding"], dtype=np.float32)




# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """
    Loads procedures.json, splits each section body into sub-chunks.
    Embeds them with Gemini embedding API, and stores vectors in ChromaDB collection.

    Startup path A:  Fast reconstruction of chunks from stored metadata;
    Startup path B:  split-> embed-> add to Chroma & persist.

    Retrieval:
        - Query still embedded by _embed_query()  
        - HNSW index returns the `n_results` nearest chunks to the query. 
    """

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            # split on paragraphs first, then sentences, words
            separators = ["\n\n", "\n", ". ", " ", ""],
        )

        # No in-memory chunk list: chunk text and metadata live entirely in the
        # ChromaDB collection. retrieve() reconstructs Chunk objects on the fly.

        ### Vector DB setup:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path = str(CHROMA_PATH))

        self._collection = self._client.get_or_create_collection(
            name = COLLECTION,
            metadata = {"hnsw:space":"cosine"}
        )
        if self._collection.count() > 0:
            self._load_from_store() 
        else:
            self._build_and_persist()


    # ---------------------------------------------------------------------------
    # Path A. Collection populated with vectors
    # ---------------------------------------------------------------------------
    def _load_from_store(self) -> None:
        # Chunk text and metadata are already in the ChromaDB collection.
        # retrieve() queries the collection directly — nothing to load into memory.
        logger.info(
            "Chroma collection '%s' already contains %d documents — "
            "skipping re-embedding. Retriever ready.",
            COLLECTION, self._collection.count(),
        )
    

    # ---------------------------------------------------------------------------
    # Path B. Collection empty: build and persist
    # ---------------------------------------------------------------------------
    def _build_and_persist(self) -> None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Data file not found at {DATA_PATH}. "
                "Please run `python src/scraper.py` first."
            )
        
        with open(DATA_PATH, encoding='utf-8') as f:
            procedures: List[dict] = json.load(f)

        # 1. Chunk each section 
        chunks: List[Chunk] = []
        # chunks are destroyed after _build_and_persist() completed,
        # memory released after Retriever initialisation finishes.
        for proc in procedures:
            if proc.get("error") or not proc.get("sections"):
                logger.warning("Skipping %s (no content).", proc.get("title"))
                continue

            for heading, body in proc["sections"].items():
                if not body.strip():
                    continue
                for sub in self._splitter.split_text(body):
                    chunk_text = f"{heading}\n{sub}"
                    chunks.append(
                        Chunk(
                            procedure_title=proc["title"],
                            url=proc["url"],
                            heading=heading,
                            text=chunk_text,
                        )
                    )

        if not chunks:
            logger.warning("No chunks produced from %s — collection remains empty.", DATA_PATH)
            return

        logger.info("Embedding %d chunks via %s...", len(chunks), EMBED_MODEL)
        texts = [c.text for c in chunks]

        # 2. embed chunks
        embeddings = _embed_documents(texts)  # (n_chunks, 768)

        # 3. persist to Chroma
        # - ids:        unique string IDs required by Chroma.
        # - documents:  chunk text stored alongside the vector (retrieved at query time).
        # - embeddings: pre-computed vectors supplied explicitly.
        # - metadatas:  source fields returned with each query result.
        self._collection.add(
            ids = [str(i) for i in range(len(chunks))],
            documents = texts,
            embeddings = embeddings.tolist(),
            metadatas=[
                {
                    "procedure_title": c.procedure_title,
                    "url": c.url,
                    "heading": c.heading,
                } for c in chunks
            ]
        )

        logger.info(
            "Retriever ready. %d chunks persisted to %s.",
            len(chunks), CHROMA_PATH,
        )

 

    ### Semantic search 
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        """
        Embed the query and return up to `top_k` chunks ranked by cosine similarity.
        """

        # Case: no relevant content found
        if not query.strip():
            return []

        # Embed query
        query_vec = _embed_query(query)  #(768,)
        raw = self._collection.query(
            query_embeddings = [query_vec.tolist()],
            n_results = min(top_k, self._collection.count()),
            include = ["documents", "metadatas", "distances"]
        )
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]
        documents = raw["documents"][0]

        results: List[Chunk] = []
        for dist, meta, doc in zip(distances, metadatas, documents):
            score = 1.0 - dist
            if score < MIN_SCORE:
                continue
            results.append(
                Chunk(
                    procedure_title=meta["procedure_title"],
                    url=meta["url"],
                    heading=meta["heading"],
                    text=doc,
                    score=score,
                )
            )

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