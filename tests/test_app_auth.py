import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import time

from src.app import app, request_timestamps, usage_stats
from src.agent import RAGAgent

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_state():
    """Clear rate limiting and usage tracking maps before each test."""
    request_timestamps.clear()
    usage_stats.clear()

def test_health_unauthenticated():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_unauthenticated():
    # Sending request without X-API-Key should return 403 Forbidden
    response = client.post("/api/chat", json={"query": "hello"})
    assert response.status_code == 403
    assert "Could not validate credentials" in response.json()["detail"]

def test_chat_invalid_key():
    # Sending request with an invalid key should return 403 Forbidden
    response = client.post(
        "/api/chat",
        json={"query": "hello"},
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 403

def test_chat_authenticated_success(monkeypatch):
    # Mock run_agent_stream on RAGAgent
    def mock_run_agent_stream(self, query, session_id):
        self.total_cost = 0.00015
        # Yield status and result
        yield {"event": "status", "data": "processing"}
        yield {"event": "result", "data": {"answer": "hello there"}}

    monkeypatch.setattr(RAGAgent, "run_agent_stream", mock_run_agent_stream)

    response = client.post(
        "/api/chat",
        json={"query": "hello"},
        headers={"X-API-Key": "oracle-secret-key"}
    )
    assert response.status_code == 200
    # Stream response check
    content = response.text
    assert "event: status" in content
    assert "event: result" in content
    
    # Check that usage stats were updated
    assert usage_stats["oracle-secret-key"]["total_requests"] == 1
    assert usage_stats["oracle-secret-key"]["total_cost"] == 0.00015

def test_rate_limiting(monkeypatch):
    # Mock agent stream
    def mock_run_agent_stream(self, query, session_id):
        yield {"event": "result", "data": {"answer": "hi"}}
    monkeypatch.setattr(RAGAgent, "run_agent_stream", mock_run_agent_stream)

    headers = {"X-API-Key": "oracle-secret-key"}
    
    # First 5 requests should pass
    for _ in range(5):
        response = client.post("/api/chat", json={"query": "hi"}, headers=headers)
        assert response.status_code == 200

    # 6th request should fail with 429 Too Many Requests
    response = client.post("/api/chat", json={"query": "hi"}, headers=headers)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]

def test_usage_endpoint(monkeypatch):
    # Authenticated usage query
    headers = {"X-API-Key": "oracle-secret-key"}
    
    # Usage initially 0
    response = client.get("/api/usage", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["overall"]["total_requests"] == 0
    assert data["your_key"]["total_requests"] == 0
    
    # Trigger a request to update usage
    def mock_run_agent_stream(self, query, session_id):
        self.total_cost = 0.002
        yield {"event": "result", "data": {"answer": "ok"}}
    monkeypatch.setattr(RAGAgent, "run_agent_stream", mock_run_agent_stream)

    client.post("/api/chat", json={"query": "hello"}, headers=headers)
    
    # Fetch usage again
    response = client.get("/api/usage", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["overall"]["total_requests"] == 1
    assert data["your_key"]["total_requests"] == 1
    assert data["your_key"]["total_cost"] == 0.002

    # Unauthenticated /api/usage should fail
    response = client.get("/api/usage", headers={"X-API-Key": "bad-key"})
    assert response.status_code == 403
