import math
from typing import Dict, Any, List, Optional
from sentence_transformers import CrossEncoder
from src.config import CONFIDENCE_THRESHOLD, DEFAULT_TOP_K, RERANKER_MODEL_NAME, RERANK_CANDIDATES_COUNT
from src.embedder import DocumentEmbedder
from src.vector_store import ChromaVectorStore

def sigmoid(x: float) -> float:
    """Computes sigmoid function to map cross-encoder scores to a [0, 1] range."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

try:
    from langsmith import traceable
except ImportError:
    def traceable(name: str = None, run_type: str = None):
        def decorator(func):
            import functools
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

class DocumentRetriever:
    def __init__(
        self,
        embedder: Optional[DocumentEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        reranker_model_name: str = RERANKER_MODEL_NAME,
        rerank_candidates_count: int = RERANK_CANDIDATES_COUNT
    ):
        self.embedder = embedder or DocumentEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.confidence_threshold = confidence_threshold
        self.reranker_model_name = reranker_model_name
        self.rerank_candidates_count = rerank_candidates_count
        self._reranker = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print(f"Loading CrossEncoder reranker model: {self.reranker_model_name}...")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    @traceable(name="Document Retriever", run_type="retriever")
    def retrieve(self, query: str, rewrite: bool = False, k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Embeds the query, retrieves candidates from ChromaDB, reranks them using CrossEncoder,
        selects the top-k, and verifies confidence against CONFIDENCE_THRESHOLD.
        """
        original_query = query
        if rewrite:
            from src.query_rewriter import QueryRewriter
            rewriter = QueryRewriter()
            query = rewriter.rewrite(query)
            print(f"Rewrote query: '{original_query}' -> '{query}'")

        # 1. Embed query and fetch candidates from Chroma DB (retrieve more than k for reranking)
        query_emb = self.embedder.embed_query(query)
        candidates_count = max(self.rerank_candidates_count, k)
        results = self.vector_store.query_similar(query_emb, k=candidates_count)
        
        # Check if we got any results
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return {
                "answerable": False,
                "message": "I don't know",
                "confidence": 0.0,
                "chunks": []
            }
            
        # Parse output candidates
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        # Prepare pairs for cross-encoder reranking
        pairs = [(query, doc) for doc in documents]
        
        # 2. Run Cross-Encoder Reranking
        print(f"Reranking {len(pairs)} candidates for query: '{query}'...")
        rerank_scores = self.reranker.predict(pairs)
        
        # Structure candidate chunks with sigmoid-normalized similarity scores
        scored_chunks = []
        for idx in range(len(ids)):
            raw_score = float(rerank_scores[idx])
            # Normalize to 0-1 range using sigmoid
            similarity = sigmoid(raw_score)
            
            scored_chunks.append({
                "id": ids[idx],
                "text": documents[idx],
                "source": metadatas[idx].get("source", "unknown"),
                "chunk_idx": metadatas[idx].get("chunk_idx", -1),
                "score": round(similarity, 4),
                "raw_score": round(raw_score, 4)
            })
            
        # Sort candidates by reranked similarity score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Get the top similarity score
        max_similarity = scored_chunks[0]["score"]
        
        # 3. Check against threshold
        if max_similarity < self.confidence_threshold:
            print(f"Retrieval rejected: max similarity {max_similarity} is below threshold {self.confidence_threshold}")
            return {
                "answerable": False,
                "message": "I don't know",
                "confidence": round(max_similarity, 4),
                "chunks": []
            }
            
        # Keep top k chunks
        top_k_chunks = scored_chunks[:k]
        
        return {
            "answerable": True,
            "message": "Success",
            "confidence": round(max_similarity, 4),
            "chunks": top_k_chunks
        }
