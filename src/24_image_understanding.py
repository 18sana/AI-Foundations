import os
import base64

from anthropic import Anthropic
from dotenv import load_dotenv

# ------------------------------------
# Load Environment
# ------------------------------------

load_dotenv()

client = Anthropic(
    api_key=os.getenv(
        "ANTHROPIC_API_KEY"
    )
)

# ------------------------------------
# Image Path
# ------------------------------------

IMAGE_PATH = "images/sample.png"

# ------------------------------------
# Read Image
# ------------------------------------

with open(
    IMAGE_PATH,
    "rb"
) as image_file:

    image_data = base64.b64encode(
        image_file.read()
    ).decode("utf-8")

# ------------------------------------
# Vision Request
# ------------------------------------

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=500,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image "
                        "in detail."
                    )
                }
            ]
        }
    ]
)

# ------------------------------------
# Output
# ------------------------------------

print("\n===== IMAGE DESCRIPTION =====\n")

print(
    response.content[0].text
)