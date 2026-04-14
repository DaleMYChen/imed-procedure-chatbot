"""
Extension 3 — RAGAS evaluation tracked with MLflow.

Runs a fixed evaluation set through the full RAG pipeline (retrieve → ask),
scores the outputs with three reference-free RAGAS metrics, and logs the
results to a local MLflow tracking store (mlruns/).

Reference-free metrics used (no ground-truth labels required):
  - Faithfulness                        : answer grounded in retrieved context?
  - AnswerRelevancy                     : answer pertinent to the question?
  - LLMContextPrecisionWithoutReference : retrieved chunks relevant to the question?

Usage:
    python src/evaluate.py

Prerequisites:
    pip install -r requirements-eval.txt
    GEMINI_API_KEY set in environment
    chroma_store/ already built (path A on startup — see Dockerfile / README)

View results after running:
    mlflow ui --backend-store-uri mlruns
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy imports: fail early with a clear message if eval deps are missing
# ---------------------------------------------------------------------------
try:
    import mlflow
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithoutReference,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError as exc:
    sys.exit(
        f"Missing evaluation dependency: {exc}\n"
        "Run:  pip install -r requirements-eval.txt"
    )

from src.retriever import get_retriever, TOP_K, EMBED_MODEL, CHUNK_SIZE
from src.llm import ask, GEMINI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation set — one question per procedure page, covering diverse intents
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    # Preparation intent
    "How should I prepare for a CT scan?",
    # Eligibility / criteria intent
    "Who is eligible for the national lung cancer screening program?",
    # Condition-specific intent
    "What should I do before angiography if I have kidney disease?",
    # Duration / logistics intent
    "How long does a PET scan take?",
    # Risk / safety intent
    "Are there any risks associated with cardiac MRI?",
    # Out-of-scope (tests robustness — expect low faithfulness / relevancy)
    "How do I repair a broken bike chain?",
]

MLFLOW_EXPERIMENT = "imed-rag-evaluation"
RAGAS_LLM_MODEL   = "gemini-1.5-flash"   # fast + cheap for evaluation judgements


# ---------------------------------------------------------------------------
# Step 1 — collect pipeline outputs
# ---------------------------------------------------------------------------

def _collect_samples(questions: list[str]) -> list[dict]:
    """
    Run each question through retrieve() + ask() and return a list of dicts
    in the format expected by ragas.EvaluationDataset.from_list().

    Fields (RAGAS 0.2+ naming):
      user_input        : the question
      response          : the LLM answer
      retrieved_contexts: list of chunk texts returned by retrieve()
    """
    retriever = get_retriever()
    samples = []

    for question in questions:
        logger.info("Collecting sample for: %r", question)
        chunks  = retriever.retrieve(question)
        result  = ask(question)

        samples.append({
            "user_input":         question,
            "response":           result.answer,
            "retrieved_contexts": [c.text for c in chunks],
        })

    return samples


# ---------------------------------------------------------------------------
# Step 2 — build RAGAS wrappers around Gemini
# ---------------------------------------------------------------------------

def _build_ragas_llm_and_embeddings(api_key: str):
    """
    Wrap Gemini models in the LangChain adapters that RAGAS expects.
    RAGAS uses the LLM as a judge for chatbot response.
    """
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=RAGAS_LLM_MODEL,
            google_api_key=api_key,
            temperature=0,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,     # reuse the same model as the retriever
            google_api_key=api_key,
        )
    )
    return ragas_llm, ragas_embeddings


# ---------------------------------------------------------------------------
# Step 3 — run RAGAS evaluation
# ---------------------------------------------------------------------------

def _run_ragas(
    samples: list[dict],
    ragas_llm,
    ragas_embeddings,
) -> tuple[dict[str, float], list[dict]]:
    """
    Evaluate the collected samples with three reference-free RAGAS metrics.

    Returns:
      aggregate_scores : {metric_name: mean_score}
      per_sample_scores: list of per-question score dicts (logged as artifact)
    """
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithoutReference(llm=ragas_llm),
    ]

    dataset = EvaluationDataset.from_list(samples)
    result  = evaluate(dataset=dataset, metrics=metrics)

    # Aggregate scores (mean across all samples)
    aggregate = {f"ragas_{m.name}": float(result[m.name]) for m in metrics}

    # Per-sample scores for the artifact
    scores_df    = result.to_pandas()
    per_sample   = scores_df.to_dict(orient="records")

    return aggregate, per_sample


# ---------------------------------------------------------------------------
# Step 4 — log everything to MLflow
# ---------------------------------------------------------------------------

def _log_to_mlflow(
    aggregate_scores: dict[str, float],
    per_sample_scores: list[dict],
    samples: list[dict],
) -> None:
    """
    Open an MLflow run and log:
      params   : pipeline configuration (models, chunk size, top-k)
      metrics  : aggregate RAGAS scores
      artifacts: per-question scores (JSON) + raw pipeline outputs (JSON)
    """
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        # --- params: what configuration produced these results? ---
        mlflow.log_params({
            "retriever_top_k":    TOP_K,
            "chunk_size":         CHUNK_SIZE,
            "embed_model":        EMBED_MODEL,
            "generation_model":   GEMINI_MODEL,
            "ragas_judge_model":  RAGAS_LLM_MODEL,
            "n_eval_questions":   len(samples),
        })

        # --- metrics: aggregate RAGAS scores ---
        mlflow.log_metrics(aggregate_scores)
        logger.info("RAGAS aggregate scores: %s", aggregate_scores)

        # --- artifacts: per-question breakdown ---
        mlflow.log_dict(per_sample_scores, "ragas_per_question.json")
        mlflow.log_dict(samples,           "pipeline_outputs.json")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set.")

    logger.info("Step 1/3 — collecting pipeline outputs for %d questions…", len(EVAL_QUESTIONS))
    samples = _collect_samples(EVAL_QUESTIONS)

    logger.info("Step 2/3 — running RAGAS evaluation…")
    ragas_llm, ragas_embeddings = _build_ragas_llm_and_embeddings(api_key)
    aggregate_scores, per_sample_scores = _run_ragas(samples, ragas_llm, ragas_embeddings)

    logger.info("Step 3/3 — logging to MLflow…")
    _log_to_mlflow(aggregate_scores, per_sample_scores, samples)

    print("\n=== RAGAS scores ===")
    for metric, score in aggregate_scores.items():
        print(f"  {metric:<45} {score:.4f}")
    print("\nTo explore runs:")
    print("  mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
