# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
# prompt = """
# A train travels 60 km/h for 2 hours
# and then 80 km/h for 3 hours.

# What distance did it travel?
# """
prompt = """
A train travels 60 km/h for 2 hours
and then 80 km/h for 3 hours.

Provide:

1. Facts
2. Calculations
3. Final Answer
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