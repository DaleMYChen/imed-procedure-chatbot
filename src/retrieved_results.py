"""
Prints the top-k chunks and their cosine similarity scores for a given query.

Usage: 
python src/retrieved_results.py "Am I eligible for a lung screening if i am 48? If i have a history of kidney disease, how shall i prepare for Angiography?"

python src/retrieved_results.py "Do I need to fast before a chest CT scan?"

"""

import sys
import logging
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from retriever import get_retriever

question = " ".join(sys.argv[1:]) or "How long does a PET scan take?"
retriever = get_retriever()
chunks = retriever.retrieve(question)

print(f"\nQuery: {question!r}")
print(f"Chunks retrieved: {len(chunks)}\n")

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(f"Procedure : {chunk.procedure_title}")
    print(f"Section   : {chunk.heading}")
    print(f"URL       : {chunk.url}")
    print(f"Score     : {chunk.score:.4f}")
    print(f"Text      :\n{chunk.text}")
    print()