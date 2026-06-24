import json
import uuid
from typing import Dict, Any, Optional
import chromadb
from src.config import CHROMA_CACHE_DB_DIR, CHROMA_CACHE_COLLECTION_NAME
from src.embedder import DocumentEmbedder

class SemanticCache:
    def __init__(
        self,
        persist_dir: str = str(CHROMA_CACHE_DB_DIR),
        collection_name: str = CHROMA_CACHE_COLLECTION_NAME,
        embedder: Optional[DocumentEmbedder] = None
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = embedder or DocumentEmbedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def lookup(self, query: str, threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        """
        Looks up the query in the semantic cache.
        If a match is found with similarity >= threshold, returns the cached response.
        Otherwise, returns None.
        """
        try:
            query_emb = self.embedder.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=1,
                include=["metadatas", "distances"]
            )
            
            if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                return None
                
            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            
            if similarity >= threshold:
                metadata = results["metadatas"][0][0]
                response_str = metadata.get("response")
                if response_str:
                    print(f"[CACHE HIT] Found match for query: '{query}' with similarity: {similarity:.4f}")
                    return json.loads(response_str)
        except Exception as e:
            print(f"Error checking cache: {e}")
            
        return None

    def add(self, query: str, response: Dict[str, Any]):
        """
        Adds a query and its corresponding response to the cache.
        """
        try:
            cache_id = str(uuid.uuid4())
            query_emb = self.embedder.embed_query(query)
            
            self.collection.add(
                ids=[cache_id],
                embeddings=[query_emb],
                metadatas=[{
                    "query": query,
                    "response": json.dumps(response)
                }],
                documents=[query]
            )
            print(f"[CACHE ADD] Cached new response for query: '{query}'")
        except Exception as e:
            print(f"Error adding to cache: {e}")

    def clear(self):
        """
        Clears all cached items from the collection.
        """
        try:
            existing = self.collection.get()
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
            print("[CACHE CLEAR] Semantic cache cleared.")
        except Exception as e:
            print(f"Error clearing cache: {e}")
