import os

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    token=os.getenv("HF_TOKEN")
)

prompt = input(
    "\nEnter image prompt: "
)

print("\nGenerating image...\n")

image = client.text_to_image(
    prompt,
    model="black-forest-labs/FLUX.1-schnell"
)

image.save(
    "generated_image.png"
)

print(
    "Image saved as generated_image.png"
)