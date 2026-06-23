import pytest
from fastapi.testclient import TestClient
from src.app import app, limiter

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_unauthorized_chat():
    # Attempting to post to chat without authorization header
    response = client.post("/api/chat", json={"query": "test query"})
    assert response.status_code == 401

    # Attempting with invalid credentials
    response = client.post(
        "/api/chat",
        headers={"Authorization": "Bearer invalid_key"},
        json={"query": "test query"}
    )
    assert response.status_code == 401

def test_unauthorized_usage():
    response = client.get("/api/usage")
    assert response.status_code == 401

    response = client.get("/api/usage", headers={"Authorization": "Bearer invalid_key"})
    assert response.status_code == 401

def test_authorized_endpoints_and_stats(monkeypatch):
    # Mock agent stream call to avoid actual LLM calls and speed up tests
    from unittest.mock import MagicMock
    from src.agent import RAGAgent

    original_stream = RAGAgent.run_agent_stream

    def mock_run_agent_stream(self, query, session_id="default"):
        # Set a cost value to verify tracking
        self.total_cost = 0.05
        yield {"event": "status", "data": "Processing..."}
        yield {"event": "token", "data": "Mocked response"}
        yield {"event": "result", "data": {
            "answer": "Mocked response",
            "citations": [],
            "confidence": 0.99,
            "follow_up_questions": []
        }}

    monkeypatch.setattr(RAGAgent, "run_agent_stream", mock_run_agent_stream)

    # 1. Fetch usage before
    resp_usage_before = client.get("/api/usage", headers={"Authorization": "Bearer key1"})
    assert resp_usage_before.status_code == 200
    usage_data = resp_usage_before.json()
    initial_queries = usage_data.get("total_queries", 0)
    initial_cost = usage_data.get("accumulated_cost_usd", 0.0)

    # 2. Make an authorized request
    # Need to bypass/reset rate limiter for key1 to ensure it doesn't fail
    if "key1" in limiter.buckets:
        limiter.buckets["key1"]["tokens"] = 5.0

    resp_chat = client.post(
        "/api/chat",
        headers={"Authorization": "Bearer key1"},
        json={"query": "Who crossed the heliopause in August 2012?", "session_id": "test_sess"}
    )
    assert resp_chat.status_code == 200
    # Stream the SSE response to trigger the generator completion
    content = resp_chat.content.decode()
    assert "event: result" in content

    # 3. Fetch usage after
    resp_usage_after = client.get("/api/usage", headers={"Authorization": "Bearer key1"})
    assert resp_usage_after.status_code == 200
    after_data = resp_usage_after.json()
    assert after_data["total_queries"] == initial_queries + 1
    assert after_data["accumulated_cost_usd"] == initial_cost + 0.05

def test_rate_limiting():
    # Make multiple requests rapidly with key2 to trigger rate limiter (cap=5)
    limiter.buckets["key2"] = {"tokens": 5.0, "last_updated": 0}
    
    # Send 5 requests (should succeed/pass auth check)
    # Using an empty mock for RAGAgent to keep it fast
    from unittest.mock import patch
    with patch("src.app.RAGAgent"):
        for _ in range(5):
            response = client.post(
                "/api/chat",
                headers={"Authorization": "Bearer key2"},
                json={"query": "rapid query"}
            )
            assert response.status_code == 200
            
        # The 6th request should be rate-limited
        response = client.post(
            "/api/chat",
            headers={"Authorization": "Bearer key2"},
            json={"query": "rapid query"}
        )
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
