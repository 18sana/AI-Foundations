# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

prompt = """
Answer using this format:

Thought:
What information do I need?

Action:
What tool would I use?

Observation:
What would the tool return?

Final Answer:

Question:
What is 25 × 47?
"""

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
print(response.content[0].text)