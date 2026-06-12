import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
def formatter(text):
    return text.upper()

tools = [
    {
        "name": "formatter",
        "description": "Convert text to uppercase",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string"
                }
            },
            "required": ["text"]
        }
    }
]
question = input(
    "Enter formatting request: "
)
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    tools=tools,
    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)
# print(response.content)
tool_use = response.content[0]

result = formatter(
    tool_use.input["text"]
)

print("\nFormatted Text:")
print(result)