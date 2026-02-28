#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Ensure Ollama is running and has the right model
echo "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install it from https://ollama.com/"
    exit 1
fi

echo "Ensuring Llama 3.2:3b is pulled..."
ollama pull llama3.2:3b

# 2. Setup Python Virtual Environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# Activate venv
source venv/bin/activate

# 3. Install Requirements in venv
echo "Installing dependencies..."
pip install -r requirements.txt

# 4. Run scraper if data doesn't exist
if [ ! -f "procedure_data/procedures.json" ]; then
    echo "Procedure information not found. Running scraper..."
    python scraper.py
else
    echo "Procedure information found. Skipping scrape."
fi

# 5. Start the Application
echo "=========================================="
echo "🚀 Starting FastAPI server on http://localhost:8000"
echo "👉 Go to http://localhost:8000/docs to test the API!"
echo "=========================================="
uvicorn src.main:app --host 0.0.0.0 --port 8000