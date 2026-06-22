import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from anthropic import Anthropic
from dotenv import load_dotenv

# ----------------------------------
# Load Environment
# ----------------------------------

load_dotenv()

client = Anthropic(
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    )
)

app = FastAPI()

# ----------------------------------
# SSE Generator
# ----------------------------------

def generate_stream():

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    "Explain Server-Sent Events "
                    "in simple terms."
                )
            }
        ]
    ) as stream:

        for text in stream.text_stream:

            yield f"data: {text}\n\n"

# ----------------------------------
# SSE Endpoint
# ----------------------------------

@app.get("/stream")

def stream():

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

# ----------------------------------
# Health Check
# ----------------------------------

@app.get("/")

def root():

    return {
        "message":
        "SSE Demo Running"
    }