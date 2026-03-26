#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Check GEMINI_API_KEY is set
echo "Checking environment..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY is not set."
    echo "   Export it first:  export GEMINI_API_KEY=your_key_here"
    exit 1
fi
echo "✅ GEMINI_API_KEY found."

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