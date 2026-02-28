# I-MED Procedure Chatbot

A RAG API that answers natural-language questions about I-MED Radiology imaging procedures, grounded in content scraped from [i-med.com.au](https://i-med.com.au).

---

## Approach

The system contains three stages:

**1. Scraping (`src/scraper.py`)**
Six procedure pages are scraped from i-med.com.au using `requests` and `BeautifulSoup`. Each page is parsed into sections (heading + paragraph pairs) and stored as JSON in `procedure_data/procedures.json`. A 1.5-second delay between requests ensures polite crawling.

**2. Semantic Search Retrieval (`src/retriever.py`)**
At startup, all scraped sections are split into chunks (one chunk: heading-body pair) and embedded using the `all-MiniLM-L6-v2` sentence-transformer model (~80MB, runs on CPU). Embeddings are held in memory as a numpy array. When a user submits a question, the query is embedded and compared against all stored chunk vectors using cosine similarity. Under this semantic search setup, the top-3 most similar chunks are returned. 

**3. Generation (`src/llm.py`)**
The retrieved chunks are assembled into a prompt alongside the user's question and sent to a local `llama3.2:3b` model running via Ollama. The prompt explicitly instructs the model to answer only from the provided context. The response is returned with source citations (procedure title, URL, section heading).

**Design reference:** Structure inspired by [umbertogriffo/rag-chatbot](https://github.com/umbertogriffo/rag-chatbot), simplified to remove vector DB and multi-turn chat history dependencies, keeping only what is necessary for a single-turn grounded Q&A API.

---


## Setup Instructions

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed on your machine
```
**Mac:** Download and install from https://ollama.com/download
**Linux:** curl -fsSL https://ollama.com/install.sh | sh
```
After installing, make sure Ollama is running: `ollama serve`

### One-command setup
```
chmod +x run.sh   # run once on setup
./run.sh    
```

`run.sh` will:
1. Create and activate a Python virtual environment
2. Install all dependencies from `requirements.txt`
3. Pull the `llama3.2:3b` model via Ollama (one-time ~2GB download)
4. Run the scraper to build `procedure_data/procedures.json`
5. Start the FastAPI server at `http://0.0.0.0:8000`

### Test the API
`POST  ->  Try it out  ->  Edit question string and execute`


### Docker 

```
docker build -t imed-chatbot .
docker run -p 8000:8000 imed-chatbot
```
Then try the API at: `http://0.0.0.0:8000 `.

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


