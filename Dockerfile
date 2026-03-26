# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY procedure_data/ ./procedure_data/

# If procedure_data/procedures.json doesn't exist yet, run scraper at build time
RUN if [ ! -f procedure_data/procedures.json ]; then python src/scraper.py; fi

# Expose FastAPI port
EXPOSE 8000

# Gemini API key — supply at runtime via:
#   docker run -e GEMINI_API_KEY=your_key ...
#   or Cloud Run --set-env-vars / --set-secrets
ENV GEMINI_API_KEY="AIzaSyDlksxjxYc8AFHdX1eQh-B-mBiqzkKf9Tk"
# src/ is not a package so we add it to PYTHONPATH for clean imports
ENV PYTHONPATH="/app/src"

# Run the API — src.main:app because main.py lives inside src/
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]