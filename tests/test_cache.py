import pytest
from src.cache import SemanticCache

@pytest.fixture
def cache():
    # Initialize cache using a temporary collection
    test_cache = SemanticCache(collection_name="test_semantic_cache_collection")
    test_cache.clear()
    yield test_cache
    test_cache.clear()
    try:
        test_cache.client.delete_collection("test_semantic_cache_collection")
    except Exception:
        pass

def test_cache_miss(cache):
    # Verify cache miss on fresh query
    res = cache.lookup("What is CAP theorem?")
    assert res is None

def test_cache_hit_exact(cache):
    # Verify cache hit on identical query
    query = "What guarantees does the CAP theorem provide?"
    response = {
        "answer": "Consistency, Availability, and Partition Tolerance.",
        "citations": ["doc_04_cap_theorem.txt"],
        "confidence": 0.99,
        "follow_up_questions": []
    }
    
    cache.add(query, response)
    
    res = cache.lookup(query)
    assert res is not None
    assert res["answer"] == response["answer"]
    assert res["confidence"] == response["confidence"]

def test_cache_hit_semantic(cache):
    # Verify cache hit on semantically near-identical query
    query_original = "What guarantees does the CAP theorem provide?"
    query_similar = "What are the CAP theorem guarantees?"
    response = {
        "answer": "Consistency, Availability, and Partition Tolerance.",
        "citations": ["doc_04_cap_theorem.txt"],
        "confidence": 0.99,
        "follow_up_questions": []
    }
    
    cache.add(query_original, response)
    
    # query_similar should be semantically close enough to trigger a hit (similarity >= 0.95)
    res = cache.lookup(query_similar, threshold=0.92)
    assert res is not None
    assert res["answer"] == response["answer"]

def test_cache_clear(cache):
    # Verify clearing cache removes everything
    query = "Who claimed quantum supremacy?"
    response = {
        "answer": "Google Sycamore processor.",
        "citations": ["doc_03_quantum_supremacy.txt"],
        "confidence": 0.98,
        "follow_up_questions": []
    }
    
    cache.add(query, response)
    assert cache.lookup(query) is not None
    
    cache.clear()
    assert cache.lookup(query) is None
