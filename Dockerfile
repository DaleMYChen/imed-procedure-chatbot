# syntax=docker/dockerfile:1
# The directive above enables BuildKit secret mounts used in the index-build step.

# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Step 1: scrape procedure pages → procedure_data/procedures.json
# (no API key needed — pure HTTP scraping)
RUN mkdir -p procedure_data && python src/scraper.py

# Step 2: build ChromaDB index → chroma_store/
# The Gemini API key is mounted as a BuildKit secret: it is never written to
# any image layer and does not appear in `docker history`.
RUN --mount=type=secret,id=gemini_key \
    GEMINI_API_KEY=$(cat /run/secrets/gemini_key) \
    python -c "from src.retriever import get_retriever; get_retriever()"

# Expose FastAPI port
EXPOSE 8000

# Run the API — src.main:app because main.py lives inside src/
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]   #pure exec form can't do variable expansion
CMD exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}