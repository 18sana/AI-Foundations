import pytest
from fastapi.testclient import TestClient
from src.tools import calculator, web_search, summarise_doc
from src.agent import RAGAgent
from src.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_safe_calculator():
    assert calculator("3 * (10 + 5)") == "45"
    assert calculator("100 / 4") == "25.0"
    assert calculator("2 ** 3") == "8"
    assert "Error" in calculator("__import__('os').system('ls')")

def test_web_search_basic():
    # Since web_search has a 20% random empty results rate, we run it up to 5 times
    results = [web_search("solid state battery energy density limit") for _ in range(5)]
    non_empty = [r for r in results if "0 results" not in r]
    assert len(non_empty) > 0
    assert any("500 Wh/kg" in r for r in non_empty)

def test_summarise_doc_basic():
    summary = summarise_doc("https://nasa.gov/artemis")
    assert "sustainable human presence" in summary.lower() or "artemis" in summary.lower()

def test_agent_run_basic():
    # We patch Anthropic messages.create to test the end-to-end agent logic without real API cost
    agent = RAGAgent()
    
    # Mock retrieve
    agent.retriever.retrieve = lambda query, rewrite=False, k=3: {
        "answerable": True,
        "message": "Success",
        "confidence": 0.95,
        "chunks": [{"id": "chunk_1", "text": "Model Context Protocol is standard.", "source": "mcp.txt", "chunk_idx": 0, "score": 0.95}]
    }
    
    # Mock model completions
    def mock_create(*args, **kwargs):
        mock_resp = type("MockResponse", (), {})()
        mock_resp.usage = type("MockUsage", (), {"input_tokens": 10, "output_tokens": 10})()
        
        # If it's the critic prompt or structure prompt
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")
        prompt = messages[0]["content"] if messages else ""
        
        mock_content = type("MockContent", (), {})()
        if "JSON schema" in prompt:
            # Structuring prompt
            mock_content.text = '{"answer": "Model Context Protocol standard.", "citations": ["mcp.txt"], "confidence": 0.95, "follow_up_questions": ["What is it?"]}'
        elif "Critic layer" in prompt:
            # Critic prompt
            mock_content.text = "Model Context Protocol standard. [mcp.txt]"
        else:
            # Agent draft response
            mock_content.text = "Model Context Protocol standard. [mcp.txt]"
            
        mock_resp.content = [mock_content]
        return mock_resp
        
    agent.client.messages.create = mock_create
    
    res = agent.run_agent("Tell me about MCP", session_id="test_session")
    
    assert res["answer"] == "Model Context Protocol standard."
    assert "mcp.txt" in res["citations"]
    assert res["confidence"] == 0.95
    assert len(res["follow_up_questions"]) > 0
