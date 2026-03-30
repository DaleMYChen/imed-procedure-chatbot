# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create the data directory and populate it by running the scraper at build time
RUN mkdir -p procedure_data && python src/scraper.py

# Expose FastAPI port
EXPOSE 8000

# Run the API — src.main:app because main.py lives inside src/
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]   #pure exec form can't do variable expansion
CMD exec uvicorn src.main:app --host 0.0.0.0 --port {$PORT: -8000}