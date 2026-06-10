from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

corpus_path = Path("../data/corpus.txt")
documents = [
    line.strip()
    for line in corpus_path.read_text().splitlines()
    if line.strip()
]

#generate embeddings -> this was not happening in chroma (embedding were hidden there)
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

doc_embeddings = model.encode(
    documents
)

#Convert to NumPy Float32
doc_embeddings = np.array(
    doc_embeddings,
    dtype="float32"
)

#create index
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(doc_embeddings)
query = "How can I learn coding?"

query_embedding = model.encode(
    [query]
)

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)
distances, indices = index.search(
    query_embedding,
    k=3
)
print("\nTop Matches:\n")

for idx in indices[0]:
    print(documents[idx])