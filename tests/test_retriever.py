import pytest
from src.retriever import DocumentRetriever

@pytest.fixture(scope="module")
def retriever():
    return DocumentRetriever()

def test_retrieve_quantum_computing(retriever):
    # Check 1: Query about quantum computing
    query = "What is quantum computing?"
    result = retriever.retrieve(query, k=3)
    
    assert result["answerable"] is True
    assert result["confidence"] >= 0.55
    assert len(result["chunks"]) > 0
    
    # Verify it finds the quantum computing/supremacy document
    sources = [chunk["source"] for chunk in result["chunks"]]
    assert any("quantum" in src for src in sources)

def test_retrieve_space_missions(retriever):
    # Check 2: Query about space missions
    query = "Space missions"
    result = retriever.retrieve(query, k=3)
    
    assert result["answerable"] is True
    assert result["confidence"] >= 0.35
    assert len(result["chunks"]) > 0
    
    # Verify it retrieves Voyager or Artemis documents
    sources = [chunk["source"] for chunk in result["chunks"]]
    assert any("voyager" in src or "artemis" in src for src in sources)

def test_retrieve_nonsense_dragons(retriever):
    # Check 3: Nonsense query
    query = "How do dragons fly?"
    result = retriever.retrieve(query, k=3)
    
    # Should not be answerable and return "I don't know"
    assert result["answerable"] is False
    assert result["message"] == "I don't know"
    assert len(result["chunks"]) == 0
    # Similarity score should be relatively low (below the 0.35 threshold)
    assert result["confidence"] < 0.35
