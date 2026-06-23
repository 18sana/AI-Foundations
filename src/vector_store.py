import chromadb
from typing import List, Dict, Any, Optional
from src.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

class ChromaVectorStore:
    def __init__(self, persist_dir: str = str(CHROMA_DB_DIR), collection_name: str = CHROMA_COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # We handle embeddings ourselves via SentenceTransformers, so we set embedding_function=None
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )

    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ):
        """
        Adds text chunks with pre-computed embeddings and metadata to ChromaDB.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def get_collection_count(self) -> int:
        """
        Returns the number of items stored in the collection.
        """
        return self.collection.count()

    def reset_collection(self):
        """
        Deletes the current collection and recreates it empty.
        """
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

    def query_similar(self, query_embedding: List[float], k: int = 3) -> Dict[str, Any]:
        """
        Queries ChromaDB for top-k similar chunks using query embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        return results
