import os
import sys
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
print("API Key loaded:", api_key[:25] + "..." if api_key else "None")

client = Anthropic(api_key=api_key)
try:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print("SUCCESS!")
    print("Response text:", response.content[0].text)
except Exception as e:
    print("EXCEPTION THROWN:", type(e), str(e))
