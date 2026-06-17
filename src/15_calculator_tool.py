import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Tool
def calculator(expression):
    return str(eval(expression))

# Tool Definition
tools = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string"
                }
            },
            "required": ["expression"]
        }
    }
]

if __name__ == "__main__":
    question = input("Ask a math question: ")

    # First call
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

    # Execute tool
    tool_use = response.content[0]

    result = calculator(
        tool_use.input["expression"]
    )

    print("\nTool Result:")
    print(result)