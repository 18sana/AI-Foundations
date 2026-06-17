import pytest


@pytest.fixture
def sample_docs():
    return [
        "RAG uses retrieval",
        "Chroma is a vector database",
        "FAISS performs similarity search"
    ]


def test_doc_count(sample_docs):
    assert len(sample_docs) == 3


def test_contains_rag(sample_docs):
    assert "RAG uses retrieval" in sample_docs