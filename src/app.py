import os
import json
import time
from threading import Lock
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, Request, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.agent import RAGAgent

load_dotenv()

# Setup valid API keys
VALID_API_KEYS = set(os.getenv("VALID_API_KEYS", "").split(","))

app = FastAPI(title="Research Oracle RAG Agent API")
security = HTTPBearer()

# Token-bucket rate limiter class
class TokenBucketLimiter:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # tokens added per second
        self.capacity = capacity  # max bucket size
        self.buckets = {}
        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self.lock:
            if client_id not in self.buckets:
                self.buckets[client_id] = {"tokens": self.capacity, "last_updated": now}
            
            bucket = self.buckets[client_id]
            elapsed = now - bucket["last_updated"]
            bucket["last_updated"] = now
            
            # Replenish tokens based on elapsed time
            bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
            
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            return False

# Rate limiter: 5 requests per 60 seconds per API key
# rate = 5 / 60 = 0.0833 tokens/sec, capacity = 5.0
limiter = TokenBucketLimiter(rate=5.0 / 60.0, capacity=5.0)

# Cost and query tracking structure
class UsageTracker:
    def __init__(self):
        self.lock = Lock()
        self.usage = defaultdict(lambda: {"total_queries": 0, "accumulated_cost_usd": 0.0})

    def record_query(self, api_key: str, cost: float):
        with self.lock:
            self.usage[api_key]["total_queries"] += 1
            self.usage[api_key]["accumulated_cost_usd"] += cost

    def get_stats(self, api_key: str) -> dict:
        with self.lock:
            stats = self.usage[api_key]
            # Create a safe alias for displaying
            alias = f"key_***{api_key[-4:]}" if len(api_key) >= 4 else "key_***"
            return {
                "api_key_alias": alias,
                "total_queries": stats["total_queries"],
                "accumulated_cost_usd": round(stats["accumulated_cost_usd"], 6)
            }

tracker = UsageTracker()

async def authenticate_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validates the bearer token against VALID_API_KEYS.
    """
    token = credentials.credentials
    if token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return token

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

async def event_generator(query: str, session_id: str, api_key: str):
    """
    Generator that runs the agent stream and yields SSE formatted chunks,
    then updates the usage tracker with the final cost.
    """
    agent = RAGAgent()
    try:
        for event in agent.run_agent_stream(query, session_id):
            event_name = event["event"]
            event_data = event["data"]
            yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"
        
        # Record usage cost once successfully completed
        tracker.record_query(api_key, agent.total_cost)
    except Exception as e:
        error_info = {"error": f"Internal agent loop error: {str(e)}"}
        yield f"event: error\ndata: {json.dumps(error_info)}\n\n"

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, api_key: str = Depends(authenticate_key)):
    """
    POST endpoint to send user query and stream back SSE tokens and status updates.
    Enforces API authentication and rate limiting.
    """
    # Enforce rate limits
    if not limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 5 requests per minute."
        )

    return StreamingResponse(
        event_generator(payload.query, payload.session_id, api_key),
        media_type="text/event-stream"
    )

@app.get("/api/usage")
def usage_endpoint(api_key: str = Depends(authenticate_key)):
    """
    GET endpoint to retrieve accumulated cost and query counts for the authenticated API key.
    """
    return tracker.get_stats(api_key)

@app.get("/health")
def health():
    return {"status": "ok"}
