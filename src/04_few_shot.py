# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

prompt = """
Review: The service was excellent.
Sentiment: Positive

Review: The product arrived damaged.
Sentiment: Negative

Review: The item was average.
Sentiment: Neutral

Review: The delivery was fast and the quality was great.
Sentiment:
"""

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=50,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response.content[0].text)