import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    )
)

# ---------------------------------
# Calculator Tool
# ---------------------------------

def calculator(
    expression: str
):

    try:

        return str(
            eval(expression)
        )

    except Exception as e:

        return str(e)

# ---------------------------------
# Tool Definitions
# ---------------------------------

tools = [
    {
        "name": "calculator",
        "description":
        "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string"
                }
            },
            "required": [
                "expression"
            ]
        }
    }
]

# ---------------------------------
# User Query
# ---------------------------------

query = input(
    "\nUser Query: "
)

print(
    "\nStatus: Thinking..."
)

# ---------------------------------
# First Agent Call
# ---------------------------------

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    tools=tools,
    messages=[
        {
            "role": "user",
            "content": query
        }
    ]
)

tool_block = None

for block in response.content:

    if block.type == "tool_use":

        tool_block = block
        break

# ---------------------------------
# Tool Path
# ---------------------------------

if tool_block:

    print(
        "\nStatus: Calling Tool..."
    )

    print(
        f"Tool: {tool_block.name}"
    )

    expression = (
        tool_block.input[
            "expression"
        ]
    )

    print(
        f"Input: {expression}"
    )

    tool_result = calculator(
        expression
    )

    print(
        f"Tool Result: {tool_result}"
    )

    print(
        "\nStatus: Generating Answer..."
    )

    final_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": response.content
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id":
                        tool_block.id,
                        "content":
                        tool_result
                    }
                ]
            }
        ]
    )

    print(
        "\n===== FINAL ANSWER =====\n"
    )

    print(
        final_response
        .content[0]
        .text
    )

# ---------------------------------
# No Tool Needed
# ---------------------------------

else:

    print(
        "\n===== ANSWER =====\n"
    )

    print(
        response.content[0].text
    )