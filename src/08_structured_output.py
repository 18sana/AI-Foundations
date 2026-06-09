import os
import json
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()

# Create Anthropic client
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# -----------------------------
# Pydantic Schema
# -----------------------------
class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]


# -----------------------------
# Input Text
# -----------------------------
article_text = """
Tesla announced a new battery technology that reduces charging time by 30%.
The company expects lower manufacturing costs and increased EV adoption.
The company expects lower manufacturing costs and increased EV adoption.
Production is expected to begin next year.
"""


# -----------------------------
# Function: Call LLM
# -----------------------------
def generate_json(prompt: str):
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

    return response.content[0].text


# -----------------------------
# Initial Prompt
# -----------------------------
prompt = f"""
Return ONLY valid JSON.

Schema:
{{
    "title": string,
    "summary": string,
    "tags": string[]
}}

Text:
{article_text}
"""

# -----------------------------
# Retry Loop
# -----------------------------
max_retries = 3

for attempt in range(max_retries):

    print(f"\nAttempt {attempt + 1}")
    print("-" * 40)

    raw_output = generate_json(prompt)

    print("Raw Model Output:")
    print(raw_output)

    try:
    
        # Parse JSON
        data = json.loads(raw_output)

        # Validate using Pydantic
        article = Article(**data)

        print("\nValidation Successful")
        print("-" * 40)

        print(article.model_dump())

        break

    except json.JSONDecodeError as e:

        print("\nJSON Parsing Failed")
        print(e)

        prompt = f"""
Your previous response was NOT valid JSON.

Error:
{e}

Return ONLY valid JSON.

Schema:
{{
    "title": string,
    "summary": string,
    "tags": string[]
}}

Text:
{article_text}
"""

    except ValidationError as e:

        print("\nSchema Validation Failed")
        print(e)

        prompt = f"""
Your previous JSON failed schema validation.

Validation Errors:
{e}

Return ONLY valid JSON matching:

{{
    "title": string,
    "summary": string,
    "tags": string[]
}}

Text:
{article_text}
"""

else:
    print("\nFailed after maximum retries.")