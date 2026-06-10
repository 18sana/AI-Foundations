from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load Corpus
# ---------------------------

corpus_path = Path("../data/corpus.txt")

documents = [
    line.strip()
    for line in corpus_path.read_text().splitlines()
    if line.strip()
]

# ---------------------------
# Models To Compare
# ---------------------------

models = {
    "MiniLM": "all-MiniLM-L6-v2",
    "BGE": "BAAI/bge-small-en-v1.5"
}

# ---------------------------
# Test Queries
# ---------------------------

queries = [
    "How do I learn coding?",
    "Tell me about AI",
    "What is a vector database?",
    "Electric cars",
    "AI assistant"
]

# ---------------------------
# Evaluation
# ---------------------------

for model_name, model_id in models.items():

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    model = SentenceTransformer(model_id)

    doc_embeddings = model.encode(documents)

    for query in queries:

        print(f"\nQuery: {query}")

        query_embedding = model.encode([query])

        scores = cosine_similarity(
            query_embedding,
            doc_embeddings
        )[0]

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        print("Top 3 Results:")

        for doc, score in ranked[:3]:
            print(f"{score:.4f} | {doc}")

        print("-" * 40)