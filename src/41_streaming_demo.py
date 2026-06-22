import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    )
)

with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=500,
    messages=[
        {
            "role": "user",
            "content": "Explain MCP."
        }
    ]
) as stream:

    for text in stream.text_stream:

        print(
            text,
            end="",
            flush=True
        )

print()