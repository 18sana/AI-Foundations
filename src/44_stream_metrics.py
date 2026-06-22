import os
import time

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    )
)

start_time = time.time()

first_token_time = None

token_count = 0

print(
    "\nStreaming...\n"
)

with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=500,
    messages=[
        {
            "role": "user",
            "content":
            "Explain vector databases."
        }
    ]
) as stream:

    for chunk in stream.text_stream:

        if first_token_time is None:

            first_token_time = time.time()

        token_count += 1

        print(
            chunk,
            end="",
            flush=True
        )

end_time = time.time()

print("\n")

ttft = (
    first_token_time
    - start_time
)

generation_time = (
    end_time
    - first_token_time
)

tokens_per_second = (
    token_count
    / generation_time
)

print(
    f"\nTTFT: {ttft:.2f}s"
)

print(
    f"Tokens/sec: {tokens_per_second:.2f}"
)