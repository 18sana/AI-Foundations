from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

_MODEL_CACHE = {}

class DocumentEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        if self.model_name not in _MODEL_CACHE:
            print(f"Loading SentenceTransformer model: {self.model_name}...")
            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        return _MODEL_CACHE[self.model_name]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of texts into list of float embeddings.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Encodes a single query into a float embedding.
        """
        return self.embed_documents([query])[0]
