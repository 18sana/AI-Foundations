# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

prompt = """
Classify the sentiment of the review
as Positive, Negative, or Neutral.

Review:
"The product is amazing and arrived quickly."
"""

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=100,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response.content[0].text)