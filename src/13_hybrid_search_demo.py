from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
import numpy as np

# --------------------------
# Load Corpus
# --------------------------

corpus_path = Path("../data/corpus.txt")

documents = [
    line.strip()
    for line in corpus_path.read_text().splitlines()
    if line.strip()
]

# --------------------------
# Semantic Search Setup
# --------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

doc_embeddings = model.encode(
    documents
)

# --------------------------
# BM25 Setup
# --------------------------

tokenized_docs = [
    doc.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(
    tokenized_docs
)

# --------------------------
# Query
# --------------------------

query = input(
    "\nEnter Query: "
)

# --------------------------
# Semantic Score
# --------------------------

query_embedding = model.encode(
    [query]
)

semantic_scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)[0]

# --------------------------
# BM25 Score
# --------------------------

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(
    tokenized_query
)

# --------------------------
# Normalize Scores
# --------------------------

semantic_scores = (
    semantic_scores -
    semantic_scores.min()
) / (
    semantic_scores.max() -
    semantic_scores.min() + 1e-8
)

bm25_scores = (
    bm25_scores -
    bm25_scores.min()
) / (
    bm25_scores.max() -
    bm25_scores.min() + 1e-8
)

# --------------------------
# Hybrid Score
# --------------------------

hybrid_scores = (
    0.7 * semantic_scores +
    0.3 * bm25_scores
)

# --------------------------
# Rank Results
# --------------------------

results = list(
    zip(
        documents,
        semantic_scores,
        bm25_scores,
        hybrid_scores
    )
)

results.sort(
    key=lambda x: x[3],
    reverse=True
)

# --------------------------
# Print Results
# --------------------------

print("\nTop Results:\n")

for doc, sem, bm, hybrid in results[:5]:
    print(
        f"""
Document : {doc}
Semantic : {sem:.4f}
BM25     : {bm:.4f}
Hybrid   : {hybrid:.4f}
"""
    )