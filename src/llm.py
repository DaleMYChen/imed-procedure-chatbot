"""
Takes a natural-language question, retrieves relevant procedure chunks via
the Retriever, and generates a grounded answer through a local Ollama model.

Adapted from the rag_chatbot_app generation step in:
  https://github.com/umbertogriffo/rag-chatbot
but simplified: single-turn, no chat history, FastAPI-friendly return dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import ollama

from .retriever import Chunk, get_retriever

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 60          # seconds; ollama.chat blocks until response
MAX_CONTEXT_CHARS = 3000     # truncate combined chunk text to stay within context window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Define response structure
# ---------------------------------------------------------------------------

@dataclass
class BotResponse:
    answer: str
    sources: List[dict]      # [{"title": ..., "url": ..., "section": ...}]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, chunks: list[Chunk]) -> str:
    """
    Assemble a RAG prompt. Both question & similar chunks in raw text.
    """
    context_parts = []
    char_count = 0
    for chunk in chunks:
        if char_count + len(chunk.text) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(
            f"[Source: {chunk.procedure_title} — {chunk.heading}]\n{chunk.text}"
        )
        char_count += len(chunk.text)

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant for patients and clinic staff who want to find out about I-MED Radiology’s imaging procedures.
            Answer ONLY based on the context below. Do not use outside knowledge.
            If the answer cannot be found in the context, say exactly:
            "I don't have enough information about that in the I-MED procedure content. Please instead submit an enquiry at https://i-med.com.au/contact-us"

            Context:
            {context}

            Question: {question}

            Answer:"""


# ---------------------------------------------------------------------------
# Main Q&A function
# ---------------------------------------------------------------------------

def ask(question: str) -> BotResponse:
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks (semantic search)
      2. Build grounded prompt
      3. Call Ollama LLM
      4. Return structured response with source citations

    Error conditions:
      - Error A: No relevant chunks found (score below threshold)
      - Error B: Ollama call fails (timeout, model not loaded, connection error)
    """
    retriever = get_retriever()

    # --- Step 1: Retrieve ---------------------------------------------------
    chunks = retriever.retrieve(question)

    # Error condition A: no relevant chunks retrieved
    if not chunks:
        return BotResponse(
            answer=(
                "I couldn't find relevant information about that in the "
                "I-MED procedure pages. Try rephrasing, or ask about one of: "
                "Angiography, Cardiac Services, CT Scan, General X-Ray, "
                "Lung Screening, or PET Scan."
            ),
            sources=[],
            error="no_relevant_content",
        )

    # --- Step 2: Build prompt -----------------------------------------------
    prompt = _build_prompt(question, chunks)

    # --- Step 3: Call Ollama ------------------------------------------------
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},   # low temp: deterministic info for medical context
        )
        answer_text = response["message"]["content"].strip()

    # Error condition B: Ollama not running / model not pulled / timeout
    except ollama.ResponseError as exc:
        logger.error("Ollama response error: %s", exc)
        return BotResponse(
            answer="The language model returned an error. Please try again.",
            sources=[],
            error=f"ollama_response_error: {exc}",
        )
    except Exception as exc:
        # Covers ConnectionRefusedError (Ollama not running), etc.
        logger.error("Ollama call failed: %s", exc)
        return BotResponse(
            answer=(
                "Could not reach the language model. "
                "Make sure Ollama is running (`ollama serve`) "
                f"and that `{OLLAMA_MODEL}` is pulled."
            ),
            sources=[],
            error=f"ollama_unavailable: {exc}",
        )

    # --- Step 4: Assemble sources -------------------------------------------
    # Deduplicate by URL while preserving chunk rank order
    seen_urls: set[str] = set()
    sources = []
    for chunk in chunks:
        if chunk.url not in seen_urls:
            sources.append({
                "title": chunk.procedure_title,
                "url": chunk.url,
                "section": chunk.heading,
            })
            seen_urls.add(chunk.url)

    return BotResponse(answer=answer_text, sources=sources)


# ---------------------------------------------------------------------------
# CLI for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    question = " ".join(sys.argv[1:]) or "How should I prepare for a CT scan?"
    result = ask(question)
    print("\n=== Answer ===")
    print(result.answer)
    print("\n=== Sources ===")
    for s in result.sources:
        print(f"  • {s['title']} ({s['section']}) — {s['url']}")
    if result.error:
        print(f"\n[Error code: {result.error}]")



# python src/llm.py "How should I prepare for a CT scan?"