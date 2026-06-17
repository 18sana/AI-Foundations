import os
import pytest
from anthropic import Anthropic


@pytest.mark.slow
def test_claude_connection():

    client = Anthropic(
        api_key=os.getenv(
            "ANTHROPIC_API_KEY"
        )
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=20,
        messages=[
            {
                "role": "user",
                "content": "Say hello"
            }
        ]
    )

    text = response.content[0].text

    assert len(text) > 0