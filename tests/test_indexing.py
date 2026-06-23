import os
import shutil
from pathlib import Path
import pytest
from src.chunker import chunk_text
from src.embedder import DocumentEmbedder
from src.vector_store import ChromaVectorStore
from src.indexer import run_indexing
from src.config import BASE_DIR, CORPUS_DIR, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

def test_chunk_text_basic():
    text = "Hello world. This is a simple test document. It contains some text."
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    
    assert len(chunks) > 0
    # Every chunk should contain text, start_char, end_char
    for chunk in chunks:
        assert "text" in chunk
        assert "start_char" in chunk
        assert "end_char" in chunk
        assert len(chunk["text"]) <= 20

def test_chunk_text_boundary_splitting():
    # Chunker should try to split on space or period instead of middle of words
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=0)
    
    # Check that chunks don't cut words in half if possible
    for c in chunks:
        # Each chunk should end with a period or space, or be whole sentence
        assert c["text"].endswith(".") or c["text"].endswith(" ") or len(c["text"]) < 25

def test_embedder_dimensions():
    embedder = DocumentEmbedder()
    test_texts = ["Hello world", "RAG agents are awesome"]
    embeddings = embedder.embed_documents(test_texts)
    
    assert len(embeddings) == 2
    # all-MiniLM-L6-v2 outputs 384 dimensional vectors
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

def test_vector_store_operations():
    # Setup test collection with temporary name
    test_collection_name = "test_indexing_collection"
    vector_store = ChromaVectorStore(collection_name=test_collection_name)
    vector_store.reset_collection()
    
    ids = ["doc1", "doc2"]
    emb1 = [1.0] + [0.0] * 383
    emb2 = [0.0] * 383 + [1.0]
    embeddings = [emb1, emb2]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    documents = ["This is doc 1 content", "This is doc 2 content"]
    
    vector_store.add_chunks(ids, embeddings, metadatas, documents)
    assert vector_store.get_collection_count() == 2
    
    # Query test - closer to emb1 than emb2
    query_emb = [0.9] + [0.0] * 383
    results = vector_store.query_similar(query_emb, k=1)
    
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "doc1"
    
    # Clean up test collection
    try:
        vector_store.client.delete_collection(test_collection_name)
    except Exception:
        pass

def test_full_indexing_pipeline():
    # Run the indexer script to index the corpus
    run_indexing(reset=True)
    
    vector_store = ChromaVectorStore()
    count = vector_store.get_collection_count()
    
    # 25 files in corpus, each must produce at least 1 chunk
    assert count >= 25
    
    # Verify we can query the indexed documents
    embedder = DocumentEmbedder()
    query_text = "Voyager 1 mission heliopause"
    query_emb = embedder.embed_query(query_text)
    
    results = vector_store.query_similar(query_emb, k=3)
    assert len(results["ids"][0]) == 3
    
    # High similarity search should mention voyager docs
    sources = [meta["source"] for meta in results["metadatas"][0]]
    assert any("voyager" in s for s in sources)
