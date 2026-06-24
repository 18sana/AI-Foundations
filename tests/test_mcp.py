import pytest
import json
import asyncio
from unittest.mock import MagicMock
from src.mcp_server import mcp, search_corpus

def test_mcp_tool_registration():
    # Retrieve registered tools from the FastMCP server
    tools = asyncio.run(mcp.list_tools())
    tool_names = [tool.name for tool in tools]
    assert "search_corpus" in tool_names


def test_search_corpus_tool_execution(monkeypatch):
    # Mock DocumentRetriever's retrieve method
    mock_results = {
        "answerable": True,
        "message": "Success",
        "confidence": 0.95,
        "chunks": [
            {
                "id": "doc1_chunk0",
                "text": "The quick brown fox jumps over the lazy dog.",
                "source": "fox.txt",
                "chunk_idx": 0,
                "score": 0.95
            }
        ]
    }
    
    mock_retrieve = MagicMock(return_value=mock_results)
    
    # We patch the retrieve method on the retriever instance imported inside mcp_server.py
    from src.mcp_server import retriever
    monkeypatch.setattr(retriever, "retrieve", mock_retrieve)
    
    # Execute the tool function
    result_str = search_corpus(query="quick brown fox", rewrite=False, k=1)
    
    # Assert retriever was called correctly
    mock_retrieve.assert_called_once_with("quick brown fox", rewrite=False, k=1)
    
    # Verify JSON output structure
    data = json.loads(result_str)
    assert data["answerable"] is True
    assert data["confidence"] == 0.95
    assert data["chunks"][0]["source"] == "fox.txt"
    assert "lazy dog" in data["chunks"][0]["text"]
