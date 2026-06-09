# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
# model = SentenceTransformer(
#     "all-MiniLM-L6-v2"
# )
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)
# Corpus
documents = [
    "Python is a programming language.",
    "Dogs are loyal pets.",
    "Machine learning is a branch of AI.",
    "Cars require regular maintenance.",
    "Basketball is a popular sport."
]

# Create embeddings
doc_embeddings = model.encode(documents)

# User query
query = "How do I learn coding?"

# Embed query
query_embedding = model.encode([query])

# Compare similarities
scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)[0]

# Pair document + score
results = list(zip(documents, scores))

# Sort descending
results.sort(
    key=lambda x: x[1],
    reverse=True
)

# Print results
for doc, score in results:
    print(
        f"{score:.4f} | {doc}"
    )