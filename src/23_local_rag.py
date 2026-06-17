import sys
import requests
from pathlib import Path
import chromadb

# Set project base directory
base_dir = Path(__file__).parent.parent

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve relevant chunks from the existing Chroma DB."""
    client = chromadb.PersistentClient(path=str(base_dir / "chroma_db"))
    collection = client.get_or_create_collection("ai_foundations")
    
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    chunks = results["documents"][0] if results["documents"] else []
    return "\n".join(chunks)

def local_rag_query(query: str) -> str:
    """Execute the local RAG pipeline."""
    # 1. Retrieval
    context = retrieve_context(query)
    
    # 2. Context Injection
    prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question:
{query}
"""
    
    # 3. Send to Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    
    return response.json()["response"]

def main():
    print("=== Local RAG Assistant (Gemma 3:1B) ===")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == "quit":
            break
            
        print("\n[Retrieving from Chroma & querying Gemma...]")
        answer = local_rag_query(user_query)
        print(f"\nGemma: {answer}\n")

if __name__ == "__main__":
    main()
