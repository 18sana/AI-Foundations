import os
import json
import time
from collections import defaultdict
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, Security, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from src.agent import RAGAgent

app = FastAPI(title="Research Oracle RAG Agent API")
agent = RAGAgent()

# API Keys Configuration
ALLOWED_KEYS = {os.getenv("API_KEY", "oracle-secret-key")}
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory Usage Tracker: {api_key: {"total_requests": int, "total_cost": float}}
usage_stats = defaultdict(lambda: {"total_requests": 0, "total_cost": 0.0})

# In-memory Rate Limiting: {api_key: [timestamps]}
request_timestamps = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 5  # requests per window

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """
    Dependency to verify API Key header.
    """
    if not api_key or api_key not in ALLOWED_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials. Please provide a valid X-API-Key."
        )
    return api_key

def check_rate_limit(api_key: str):
    """
    Sliding window rate limiter per API key.
    """
    now = time.time()
    # Keep only timestamps in the current window
    request_timestamps[api_key] = [
        t for t in request_timestamps[api_key] if now - t < RATE_LIMIT_WINDOW
    ]
    if len(request_timestamps[api_key]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 5 requests per minute allowed."
        )
    request_timestamps[api_key].append(now)

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

async def event_generator(query: str, session_id: str, api_key: str):
    """
    Generator that runs the agent stream and yields SSE formatted chunks.
    Tracks query cost upon completion.
    """
    try:
        # Run agent stream
        for event in agent.run_agent_stream(query, session_id):
            event_name = event["event"]
            event_data = event["data"]
            # Format as SSE
            yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"
        
        # Record usage and cost after successful run
        cost = getattr(agent, "total_cost", 0.0)
        usage_stats[api_key]["total_requests"] += 1
        usage_stats[api_key]["total_cost"] += cost
    except Exception as e:
        error_info = {"error": f"Internal agent loop error: {str(e)}"}
        yield f"event: error\ndata: {json.dumps(error_info)}\n\n"

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, api_key: str = Depends(verify_api_key)):
    """
    POST endpoint to send user query and stream back SSE tokens and status updates.
    Protected by API key and rate limiting.
    """
    check_rate_limit(api_key)
    return StreamingResponse(
        event_generator(payload.query, payload.session_id, api_key),
        media_type="text/event-stream"
    )

@app.get("/api/usage")
def get_usage(api_key: str = Depends(verify_api_key)):
    """
    Get current usage statistics.
    """
    overall_requests = sum(stats["total_requests"] for stats in usage_stats.values())
    overall_cost = sum(stats["total_cost"] for stats in usage_stats.values())
    
    return {
        "overall": {
            "total_requests": overall_requests,
            "total_cost": round(overall_cost, 6)
        },
        "keys": {
            k: {
                "total_requests": stats["total_requests"],
                "total_cost": round(stats["total_cost"], 6)
            } for k, stats in usage_stats.items()
        },
        "your_key": {
            "key": api_key,
            "total_requests": usage_stats[api_key]["total_requests"],
            "total_cost": round(usage_stats[api_key]["total_cost"], 6)
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

