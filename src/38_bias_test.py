import os
import json

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    )
)

# ------------------------
# Load Dataset
# ------------------------

with open(
    "src/39_bias_dataset.json",
    "r"
) as f:

    test_cases = json.load(f)

results = []

# ------------------------
# Run Tests
# ------------------------

for case in test_cases:

    print(
        f"\nRunning Test {case['id']}"
    )

    response_a = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": case["prompt_a"]
            }
        ]
    )

    response_b = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": case["prompt_b"]
            }
        ]
    )

    output_a = response_a.content[0].text
    output_b = response_b.content[0].text

    results.append(
        {
            "id": case["id"],
            "prompt_a": case["prompt_a"],
            "prompt_b": case["prompt_b"],
            "output_a": output_a,
            "output_b": output_b
        }
    )

# ------------------------
# Save Results
# ------------------------

with open(
    "bias_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )

print(
    "\nResults saved to bias_results.json"
)