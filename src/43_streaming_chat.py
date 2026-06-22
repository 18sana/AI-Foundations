import os
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

from anthropic import Anthropic
from dotenv import load_dotenv

from pydantic import BaseModel
from pydantic import ValidationError

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
# Schema
# ----------------------------------

class ResponseSchema(BaseModel):

    topic: str
    summary: str
    confidence: float

# ----------------------------------
# Home Page
# ----------------------------------

@app.get("/")
def home():

    with open(
        "templates/index.html",
        "r"
    ) as f:

        return HTMLResponse(
            f.read()
        )

# ----------------------------------
# Streaming Endpoint
# ----------------------------------

@app.get("/stream")
def stream(query: str):

    def generate():

        complete_text = ""

        yield (
            "event: status\n"
            "data: Streaming started\n\n"
        )

        try:

            with client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=500,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content":
                        f"""
Respond ONLY with valid JSON.

DO NOT use markdown.
DO NOT use code fences.
DO NOT add explanations.

Return exactly this schema:

{{
    "topic": "string",
    "summary": "string",
    "confidence": 0.95
}}

Query:
{query}
"""
                    }
                ]
            ) as stream:

                for chunk in stream.text_stream:

                    complete_text += chunk

                    yield (
                        f"event: token\n"
                        f"data: {chunk}\n\n"
                    )

            yield (
                "event: status\n"
                "data: Validating JSON\n\n"
            )

            print("\n")
            print("=" * 60)
            print("COMPLETE RESPONSE")
            print("=" * 60)
            print(complete_text)
            print("=" * 60)

            complete_text = complete_text.strip()

            if complete_text.startswith(
                "```json"
            ):
                complete_text = complete_text.replace(
                    "```json",
                    ""
                )

            if complete_text.startswith(
                "```"
            ):
                complete_text = complete_text.replace(
                    "```",
                    ""
                )

            if complete_text.endswith(
                "```"
            ):
                complete_text = complete_text[:-3]

            complete_text = complete_text.strip()

            parsed = json.loads(
                complete_text
            )

            validated = ResponseSchema(
                **parsed
            )

            yield (
                "event: structured\n"
                f"data: {validated.model_dump_json()}\n\n"
            )

            yield (
                "event: complete\n"
                "data: done\n\n"
            )

        except ValidationError as e:

            yield (
                "event: error\n"
                f"data: Schema Validation Failed: {str(e)}\n\n"
            )

        except Exception as e:

            yield (
                "event: error\n"
                f"data: {str(e)}\n\n"
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )