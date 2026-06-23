import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DB_DIR = BASE_DIR / "chroma_db" / "documents"
CHROMA_MEMORY_DB_DIR = BASE_DIR / "chroma_db" / "memory"

# Embedding settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB settings
CHROMA_COLLECTION_NAME = "document_corpus"
CHROMA_MEMORY_COLLECTION_NAME = "conversation_memory"
CONFIDENCE_THRESHOLD = 0.35
DEFAULT_TOP_K = 3
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES_COUNT = 10

# Chunking settings
CHUNK_SIZE = 500  # Characters
CHUNK_OVERLAP = 50  # Characters

# Ensure directories exist
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_MEMORY_DB_DIR.mkdir(parents=True, exist_ok=True)
