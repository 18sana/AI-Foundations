# import os
# import base64

# from anthropic import Anthropic
# from dotenv import load_dotenv

# # ------------------------------------
# # Load Environment
# # ------------------------------------

# load_dotenv()

# client = Anthropic(
#     api_key=os.getenv(
#         "ANTHROPIC_API_KEY"
#     )
# )

# # ------------------------------------
# # Image Path
# # ------------------------------------

# IMAGE_PATH = "images/sample.png"

# # ------------------------------------
# # Read Image
# # ------------------------------------

# with open(
#     IMAGE_PATH,
#     "rb"
# ) as image_file:

#     image_data = base64.b64encode(
#         image_file.read()
#     ).decode("utf-8")

# # ------------------------------------
# # Vision Request
# # ------------------------------------

# response = client.messages.create(
#     model="claude-haiku-4-5",
#     max_tokens=500,
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": "image/png",
#                         "data": image_data
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": (
#                         "Describe this image "
#                         "in detail."
#                     )
#                 }
#             ]
#         }
#     ]
# )

# # ------------------------------------
# # Output
# # ------------------------------------

# print("\n===== IMAGE DESCRIPTION =====\n")

# print(
#     response.content[0].text
# )


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
# Reusable Function
# ------------------------------------

def describe_image(
    image_path: str,
    prompt: str = "Describe this image in detail."
):

    # ------------------------------------
    # Detect Media Type
    # ------------------------------------

    if image_path.lower().endswith(".png"):
        media_type = "image/png"

    elif image_path.lower().endswith(".jpg"):
        media_type = "image/jpeg"

    elif image_path.lower().endswith(".jpeg"):
        media_type = "image/jpeg"

    else:
        raise ValueError(
            f"Unsupported image type: {image_path}"
        )

    # ------------------------------------
    # Read Image
    # ------------------------------------

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    print(
        f"DEBUG: {image_path} -> {media_type}"
    )

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
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    return response.content[0].text
# ------------------------------------
# Standalone Testing
# ------------------------------------

if __name__ == "__main__":

    IMAGE_PATH = "images/sample.png"

    result = describe_image(
        IMAGE_PATH
    )

    print(
        "\n===== IMAGE DESCRIPTION =====\n"
    )

    print(result)