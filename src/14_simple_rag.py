from pathlib import Path
import os

import chromadb
# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

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
# Chroma Setup
# ---------------------------

chroma_client = chromadb.PersistentClient(
    path="../chroma_db"
)

# Fresh collection every run

try:
    chroma_client.delete_collection(
        "rag_demo"
    )
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="rag_demo"
)

collection.add(
    ids=[str(i) for i in range(len(documents))],
    documents=documents
)

# ---------------------------
# User Question
# ---------------------------

question = input(
    "\nAsk a question: "
)

# ---------------------------
# Retrieve
# ---------------------------

results = collection.query(
    query_texts=[question],
    n_results=3
)

retrieved_docs = results["documents"][0]

# ---------------------------
# Build Context
# ---------------------------

context = "\n".join(
    retrieved_docs
)

print("\nRetrieved Context:\n")
print(context)

# ---------------------------
# Generate Answer
# ---------------------------

prompt = f"""
Answer the question using ONLY the provided context.

If the answer is not in the context,
say:

"I could not find that information
in the provided documents."

Context:
{context}

Question:
{question}
"""

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAnswer:\n")
print(response.content[0].text)