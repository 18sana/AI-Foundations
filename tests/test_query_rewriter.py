import os
import pytest
from unittest.mock import MagicMock, patch
from src.query_rewriter import QueryRewriter

@pytest.fixture(scope="module")
def rewriter():
    return QueryRewriter()

def test_rewrite_mcp(rewriter):
    # Test 1: Input: Tell me about MCP -> Output contains: Model Context Protocol
    query = "Tell me about MCP"
    rewritten = rewriter.rewrite(query)
    print(f"Original: {query} | Rewritten: {rewritten}")
    
    assert "model context protocol" in rewritten.lower()

def test_rewrite_chroma(rewriter):
    # Test 2: Input: How does Chroma work? -> Output contains Vector Database, Embeddings, or Similarity Search
    query = "How does Chroma work?"
    rewritten = rewriter.rewrite(query)
    print(f"Original: {query} | Rewritten: {rewritten}")
    
    lower_rewritten = rewritten.lower()
    matches = ["vector database", "embedding", "similarity search", "chromadb", "vector store"]
    assert any(term in lower_rewritten for term in matches)

def test_rewrite_is_search_query_not_answer(rewriter):
    # Test 3: Output should be a search query, not a conversational answer.
    query = "What is the Voyager mission?"
    rewritten = rewriter.rewrite(query)
    print(f"Original: {query} | Rewritten: {rewritten}")
    
    # It should not behave like a chatbot answering the question or giving a conversational explanation.
    lower_rewritten = rewritten.lower()
    conversational_phrases = ["here is", "sure, i can", "the voyager mission is", "voyager 1 was", "according to"]
    for phrase in conversational_phrases:
        assert phrase not in lower_rewritten
    
    # It should look like keyword tags or search phrases
    assert len(rewritten.split()) <= 20
