import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

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
goal = input(
    "\nEnter Goal: "
)
MAX_STEPS = 5

messages = [
    {
        "role": "user",
        "content": goal
    }
]
for step in range(MAX_STEPS):

    print(f"\n--- Step {step + 1} ---")

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        tools=tools,
        messages=messages
    )
tool_use = None

for block in response.content:
    if block.type == "tool_use":
        tool_use = block
        if not tool_use:

    print("\nFinal Answer:\n")

    print(response.content[0].text)

    break
print(
    f"Using Tool: {tool_use.name}"
)

result = calculator(
    tool_use.input["expression"]
)

print(
    f"Observation: {result}"
)
messages.append(
    {
        "role": "assistant",
        "content": response.content
    }
)

messages.append(
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result
            }
        ]
    }
)
