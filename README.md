# I-MED Procedure Chatbot

A RAG API that answers natural-language questions about I-MED Radiology imaging procedures, grounded in content scraped from [i-med.com.au](https://i-med.com.au).

---

## Approach

The system contains three stages:

**1. Scraping (`src/scraper.py`)**
Six procedure pages are scraped from i-med.com.au using `requests` and `BeautifulSoup`. Each page is parsed into sections (heading + paragraph pairs) and stored as JSON in `procedure_data/procedures.json`. A 1.5-second delay between requests ensures polite crawling.

**2. Semantic Search Retrieval (`src/retriever.py`)**
Each procedure section is sub-chunked using LangChain's `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50) to prevent long bodies from overflowing the embedding token limit. Chunks are embedded with Google's `gemini-embedding-001` model (768-dim, via Gemini API) and stored in a ChromaDB persistent collection (`chroma_store/`) backed by an HNSW index. The index is built once at Docker image build time and reused on every container start. At query time the question is embedded and the top-3 most similar chunks are retrieved by cosine similarity.

**3. Generation (`src/llm.py`)**
The retrieved chunks are assembled into a prompt alongside the user's question and sent to the Gemini API (`gemini-2.0-flash`). The prompt explicitly instructs the model to answer only from the provided context. The response is returned with source citations (procedure title, URL, section heading).

**4. Evaluation (`src/evaluate.py`)**
A fixed evaluation set is run through the full pipeline and scored with three reference-free [RAGAS](https://docs.ragas.io) metrics — Faithfulness, Answer Relevancy, and Context Precision — using `gemini-1.5-flash` as the evaluator LLM. Results are logged to a local [MLflow](https://mlflow.org) tracking store (`mlruns/`) for run-over-run comparison.

**Design reference:** Structure inspired by [umbertogriffo/rag-chatbot](https://github.com/umbertogriffo/rag-chatbot), simplified to single-turn grounded Q&A with a persistent ChromaDB vector store.

---


## Setup Instructions

### Prerequisites

- Python 3.9+
- A Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com)

```bash
export GEMINI_API_KEY=your_key_here
```

### One-command setup
```bash
chmod +x run.sh   # run once on setup
./run.sh
```

`run.sh` will:
1. Check `GEMINI_API_KEY` is set
2. Create and activate a Python virtual environment
3. Install all dependencies from `requirements.txt`
4. Run the scraper to build `procedure_data/procedures.json`
5. Start the FastAPI server at `http://0.0.0.0:8000`

The ChromaDB vector index (`chroma_store/`) is built automatically on first run and reused on subsequent starts.

### Test the API
`POST  ->  Try it out  ->  Edit question string and execute`


### Docker

The Gemini API key is passed as a [BuildKit secret](https://docs.docker.com/build/secrets/) — it is used only during the index-build step and is never written into any image layer.

```bash
export GEMINI_API_KEY=your_key_here
docker build --secret id=gemini_key,env=GEMINI_API_KEY -t imed-chatbot .
docker run -e GEMINI_API_KEY=$GEMINI_API_KEY -p 8000:8000 imed-chatbot
```
Then try the API at `http://0.0.0.0:8000`.

### Evaluation (optional)

Runs the full pipeline against a fixed question set and logs [RAGAS](https://docs.ragas.io) scores to a local MLflow store.

```bash
pip install -r requirements-eval.txt
python src/evaluate.py
mlflow ui --backend-store-uri mlruns   # view results at http://localhost:5000
```

### GCP Cloud Run
```
https://imed-chatbot-1053993616231.australia-southeast1.run.app
```

---

## Example Queries and Outputs

### Query 1 

**Input:**
```json
{ "question": "How should I prepare for a CT scan?" }
```

**Output:**
```json
{
  "answer": "To prepare for a CT scan, you may need to:\n\n* Remove any metal objects such as jewelry, glasses, or clothing with metal fasteners\n* Wear a hospital gown and remove any makeup, hairpins, or other items that could interfere with the scan\n* Inform your doctor or radiographer of any medical implants, pacemakers, or other devices that may be affected by the scan\n* Follow any specific instructions provided by your doctor or radiographer\n\nPlease note that these are general guidelines and it's best to follow the specific instructions provided by your healthcare team.",
  "sources": [
    {
      "title": "CT Scan",
      "url": "https://i-med.com.au/procedures/ct-scan",
      "section": "What is a CT scan?"
    }
  ],
  "error": null
}
```

---

### Query 2 

**Input:**
```json
{ "question": "Am I eligible for a lung screening if I am 48?" }
```

**Output:**
```json
{
  "answer": "No, you are not eligible for the National Lung Cancer Screening Program (NLCSP) at this age. According to the eligibility criteria, you need to be between 50 and 70 years old to be eligible for the program.",
  "sources": [
    {
      "title": "Lung Screening",
      "url": "https://i-med.com.au/procedures/lung-screening",
      "section": "Who is eligible for the screening program?"
    }
  ],
  "error": null
}
```

---

### Query 3 

**Input:**
```json
{ "question": "If I have kidney disease, how should I prepare for Angiography?" }
```

**Output:**
```json
{
  "answer": "If you have kidney disease, you must inform staff when you make the appointment and when you attend for the procedure. This ensures that the appropriate measures are taken to carry out the procedure with maximum safety.",
  "sources": [
    {
      "title": "Angiography",
      "url": "https://i-med.com.au/procedures/angiography",
      "section": "How do I prepare for angiography? keyboard_arrow_down"
    }
  ],
  "error": null
}
```

---

### Query 4 

**Input:**
```json
{ "question": "How long does a PET scan take?" }
```

**Output:**
```json
{
  "answer": "You can expect to be in the PET imaging department for between two to three hours. The time on the PET scanner is typically 30 minutes but time is also needed for preparation.",
  "sources": [
    {
      "title": "PET Scan",
      "url": "https://i-med.com.au/procedures/pet-scan",
      "section": "How long does a PET scan take? keyboard_arrow_down"
    },
    {
      "title": "Lung Screening",
      "url": "https://i-med.com.au/procedures/lung-screening",
      "section": "What does the lung screening scan involve?"
    }
  ],
  "error": null
}
```

---

### Query 5 — Out-of-scope question (error handling)

**Input:**
```json
{ "question": "How do I repair a broken bike chain?" }
```

**Output:**
```json
{
  "answer": "I couldn't find relevant information about that in the I-MED procedure pages. Try rephrasing, or ask about one of: Angiography, Cardiac Services, CT Scan, General X-Ray, Lung Screening, or PET Scan.",
  "sources": [],
  "error": "no_relevant_content"
}
```

---

## Known Limitations

**Scraped contents are fixed at build time.** The chatbot only covers 6 procedure pages. Any content added or updated on i-med.com.au after scraping will not be reflected until the scraper is re-run.

**JavaScript-rendered pages cannot be scraped.** The scraper uses `requests` + `BeautifulSoup` which only reads static HTML content. Pages that load content dynamically after users enter input, such as the clinic finder at `/find-a-radiology-clinic`, return only an empty shell. 

**Queries with mixed in-scope and out-of-scope sub-questions will not be answered separately.** When a single prompt contains multiple sub-questions and at least one falls outside the scraped content, the LLM tends to apply a blanket "insufficient information" response across all sub-questions rather than answering each independently. This is a prompt architecture limitation: all sub-questions share a single context block and a single `TOP_K=3` retrieval budget. 
<br>
A production fix would decompose compound queries before retrieval and run separate RAG passes per sub-question.

**Operational information is out of scope.** The public i-med.com.au website is essentially a brochure: it does not contain booking details, pricing, scan results, or clinic-specific contacts. These are handled by internal I-MED systems. The chatbot cannot advise on questions that can lead to personal information being shared. 


