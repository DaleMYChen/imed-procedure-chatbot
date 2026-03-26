# Define scripts as a web service using FastAPI

import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .llm import ask, BotResponse

app = FastAPI(title="I-MED Procedure Assistant")


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


@app.post("/api/ask", response_model=AnswerResponse)
def ask_question(request: QueryRequest):
    """
    Ask a natural-language question about I-MED imaging procedures.
    Returns a grounded answer and the source page(s) it was drawn from.
    """
    result = ask(request.question)

    # Surface Gemini-level failures as 503 so the caller knows to retry
    if result.error and "gemini" in result.error:
        raise HTTPException(status_code=503, detail=result.answer)

    return AnswerResponse(
        answer=result.answer,
        sources=[SourceSchema(**s) for s in result.sources],
        error=result.error,
    )