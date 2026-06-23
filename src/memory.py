import uuid
import time
from typing import List, Dict, Any, Optional
import chromadb
from src.config import CHROMA_MEMORY_DB_DIR, CHROMA_MEMORY_COLLECTION_NAME
from src.embedder import DocumentEmbedder

class SemanticMemory:
    def __init__(
        self,
        persist_dir: str = str(CHROMA_MEMORY_DB_DIR),
        collection_name: str = CHROMA_MEMORY_COLLECTION_NAME,
        embedder: Optional[DocumentEmbedder] = None
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = embedder or DocumentEmbedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def save_exchange(self, session_id: str, user: str, assistant: str) -> str:
        """
        Saves a single conversation exchange (user input and assistant output)
        into ChromaDB with metadata and pre-computed embedding.
        Returns the exchange ID.
        """
        exchange_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Serialize the exchange for embedding and storage
        text_content = f"User: {user}\nAssistant: {assistant}"
        embed_content = f"{user}\n{assistant}"
        
        # Compute embedding
        embedding = self.embedder.embed_query(embed_content)
        
        # Add to Chroma DB
        self.collection.add(
            ids=[exchange_id],
            embeddings=[embedding],
            metadatas=[{
                "session_id": session_id,
                "timestamp": timestamp,
                "user": user,
                "assistant": assistant
            }],
            documents=[text_content]
        )
        return exchange_id

    def retrieve_memories(self, session_id: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most semantically relevant prior exchanges
        for the given session_id.
        """
        # Embed current query
        query_emb = self.embedder.embed_query(query)
        
        # Query ChromaDB collection, filtering by session_id
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            where={"session_id": session_id},
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return []
            
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        memories = []
        for idx in range(len(ids)):
            similarity = 1.0 - distances[idx]
            memories.append({
                "id": ids[idx],
                "text": documents[idx],
                "user": metadatas[idx].get("user", ""),
                "assistant": metadatas[idx].get("assistant", ""),
                "timestamp": metadatas[idx].get("timestamp", 0.0),
                "score": round(similarity, 4)
            })
            
        # Sort by similarity score descending (or timestamp if required, but semantic query returns best match first)
        return memories

    def reset_memories(self):
        """
        Clears the memory collection.
        """
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
