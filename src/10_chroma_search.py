# pyrefly: ignore [missing-import]
from pathlib import Path
import chromadb
corpus_path = Path("../data/corpus.txt")

documents = [
    line.strip()
    for line in corpus_path.read_text().splitlines()
    if line.strip()
]

client = chromadb.PersistentClient(
    path="../chroma_db"
)
collection = client.get_or_create_collection(
    name="ai_foundations"
)
# This recreates the collection every run.
# try:
#     client.delete_collection(
#         "ai_foundations"
#     )
# except:
#     pass

# collection = client.get_or_create_collection(
#     name="ai_foundations"
# )

#documents -> embeddings -> store vector
collection.add(
    ids=[str(i) for i in range(len(documents))],
    documents=documents
)
#embeddings-> similarity search -> top-results
results = collection.query(
    query_texts=["How can I learn coding?"],
    n_results=3
)

# print(results)
print("\nTop Matches:\n")

for doc in results["documents"][0]:
    print(doc)


