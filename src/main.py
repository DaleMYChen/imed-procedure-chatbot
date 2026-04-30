# Define scripts as a web service using FastAPI

import json
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.llm import ask, BotResponse

app = FastAPI(title="I-MED Procedure Assistant")
logger = logging.getLogger(__name__)

### 1. Define pydantic dataclasses to validate
class QueryRequest(BaseModel):
    question: str


class SourceSchema(BaseModel):
    title: str
    url: str
    section: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceSchema]
    error: Optional[str] = None


### 2. Define routes

# a landing page
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head><title>I-MED Bot API</title></head>
        <body style="font-family: Arial, sans-serif; padding: 2rem;">
            <h1>I-MED Procedure Assistant is Running! 🚀</h1>
            <p>Go to <a href="/docs">/docs</a> to test the interactive web UI.</p>
        </body>
    </html>
    """


# the request-response endpoint
@app.post("/api/ask", response_model=AnswerResponse)
def ask_question(request: QueryRequest):
    """
    Input request to /api/ask is a str question
    """
    t0 = time.perf_counter()
    result = ask(request.question)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(json.dumps({
        "event": "ask",
        "latency_ms": latency_ms,
        "sources_returned": len(result.sources),
        "error": result.error,
    }))

    # Surface Gemini-level failures as 503 so the caller knows to retry
    if result.error and "gemini" in result.error:
        raise HTTPException(status_code=503, detail=result.answer)

    return AnswerResponse(
        answer=result.answer,
        sources=[SourceSchema(**s) for s in result.sources],
        error=result.error,
    )