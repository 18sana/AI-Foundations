# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

text = """
Tesla announced a new battery technology that reduces charging time by 30%.
Production starts next year and manufacturing costs are expected to decrease.
"""

# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=200,
#     messages=[
#         {
#             "role": "user",
#             "content": f"Summarize this:\n{text}"
#         }
#     ]
# )

# print(response.content[0].text)

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    messages=[
        {
            "role": "user",
            "content": f"""
Summarize the text using this format:

Company:
Technology:
Benefits:
Timeline:

Text:
{text}
"""
        }
    ]
)
print(response.content[0].text)

