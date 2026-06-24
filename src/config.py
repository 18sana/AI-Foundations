import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith Configurations
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-foundations")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

if LANGCHAIN_TRACING_V2 and not LANGCHAIN_API_KEY:
    print("Warning: LANGCHAIN_TRACING_V2 is enabled, but LANGCHAIN_API_KEY is not set. Traces will not be sent to LangSmith.")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DB_DIR = BASE_DIR / "chroma_db" / "documents"
CHROMA_MEMORY_DB_DIR = BASE_DIR / "chroma_db" / "memory"
CHROMA_CACHE_DB_DIR = BASE_DIR / "chroma_db" / "cache"

# Embedding settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB settings
CHROMA_COLLECTION_NAME = "document_corpus"
CHROMA_MEMORY_COLLECTION_NAME = "conversation_memory"
CHROMA_CACHE_COLLECTION_NAME = "semantic_cache"
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
CHROMA_CACHE_DB_DIR.mkdir(parents=True, exist_ok=True)
