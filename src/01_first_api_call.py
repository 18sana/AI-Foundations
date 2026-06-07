# pyrefly: ignore [missing-import]
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

response = client.messages.create(
    model="claude-sonnet-4-0",
    max_tokens=30,
    messages=[
        {
            "role": "user",
            "content": "What is Machine Learning?"
        }
    ]
)

print(response.content[0].text)
print(response.usage)