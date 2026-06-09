# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

prompt = """
A farmer has 15 cows.
5 are sold.
3 are born.

How many cows does he have now?
"""

for i in range(5):
    response = client.messages.create(
        model="claude-haiku-4-5",
        temperature=0.8,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(f"Run {i+1}")
    print(response.content[0].text)
    print("-" * 40)