from typing import Dict, Any, List, Optional
from src.config import CONFIDENCE_THRESHOLD, DEFAULT_TOP_K
from src.embedder import DocumentEmbedder
from src.vector_store import ChromaVectorStore

class DocumentRetriever:
    def __init__(
        self,
        embedder: Optional[DocumentEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        self.embedder = embedder or DocumentEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.confidence_threshold = confidence_threshold

    def retrieve(self, query: str, rewrite: bool = False, k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Embeds the query, queries ChromaDB, and returns the top-k chunks
        with their calculated confidence scores.
        Rejects the query if the maximum confidence is below the threshold.
        """
        original_query = query
        if rewrite:
            from src.query_rewriter import QueryRewriter
            rewriter = QueryRewriter()
            query = rewriter.rewrite(query)
            print(f"Rewrote query: '{original_query}' -> '{query}'")

        # 1. Embed query
        query_emb = self.embedder.embed_query(query)
        
        # 2. Query Chroma DB
        results = self.vector_store.query_similar(query_emb, k=k)
        
        # Check if we got any results
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return {
                "answerable": False,
                "message": "I don't know",
                "confidence": 0.0,
                "chunks": []
            }
            
        # Parse output
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        chunks = []
        max_similarity = 0.0
        
        for idx in range(len(ids)):
            # Convert cosine distance (Chroma HNSW default space is cosine distance: 1 - cosine_sim)
            # Similarity = 1.0 - distance
            similarity = 1.0 - distances[idx]
            max_similarity = max(max_similarity, similarity)
            
            chunks.append({
                "id": ids[idx],
                "text": documents[idx],
                "source": metadatas[idx].get("source", "unknown"),
                "chunk_idx": metadatas[idx].get("chunk_idx", -1),
                "score": round(similarity, 4)
            })
            
        # 3. Check against threshold
        if max_similarity < self.confidence_threshold:
            return {
                "answerable": False,
                "message": "I don't know",
                "confidence": round(max_similarity, 4),
                "chunks": []
            }
            
        return {
            "answerable": True,
            "message": "Success",
            "confidence": round(max_similarity, 4),
            "chunks": chunks
        }
