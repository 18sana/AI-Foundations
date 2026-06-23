import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.agent import RAGAgent

app = FastAPI(title="Research Oracle RAG Agent API")
agent = RAGAgent()

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

async def event_generator(query: str, session_id: str):
    """
    Generator that runs the agent stream and yields SSE formatted chunks.
    """
    try:
        # Run agent stream
        for event in agent.run_agent_stream(query, session_id):
            event_name = event["event"]
            event_data = event["data"]
            # Format as SSE
            yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"
    except Exception as e:
        error_info = {"error": f"Internal agent loop error: {str(e)}"}
        yield f"event: error\ndata: {json.dumps(error_info)}\n\n"

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    POST endpoint to send user query and stream back SSE tokens and status updates.
    """
    return StreamingResponse(
        event_generator(payload.query, payload.session_id),
        media_type="text/event-stream"
    )

@app.get("/health")
def health():
    return {"status": "ok"}
