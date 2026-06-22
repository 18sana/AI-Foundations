import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    )
)

query = input(
    "\nUser Query: "
)

# ----------------------------------
# Step 1 Draft
# ----------------------------------

draft = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    messages=[
        {
            "role": "user",
            "content": query
        }
    ]
)

draft_answer = (
    draft.content[0].text
)

print(
    "\n===== DRAFT =====\n"
)

print(
    draft_answer
)

# ----------------------------------
# Step 2 Critique
# ----------------------------------

critique = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    messages=[
        {
            "role": "user",
            "content":
            f"""
Review this answer.

Answer:
{draft_answer}

Check for:
- Harmful content
- Incorrect claims
- Bias
- Missing context

Provide critique.
"""
        }
    ]
)

critique_text = (
    critique.content[0].text
)

print(
    "\n===== CRITIQUE =====\n"
)

print(
    critique_text
)

# ----------------------------------
# Step 3 Revision
# ----------------------------------

revision = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    messages=[
        {
            "role": "user",
            "content":
            f"""
Original Answer:
{draft_answer}

Critique:
{critique_text}

Improve the answer.
"""
        }
    ]
)

print(
    "\n===== FINAL ANSWER =====\n"
)

print(
    revision.content[0].text
)