import os
import importlib.util

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
# Dynamically Import Image Understanding
# ------------------------------------

spec = importlib.util.spec_from_file_location(
    "image_understanding",
    "src/24_image_understanding.py"
)

image_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    image_module
)

describe_image = (
    image_module.describe_image
)

# ------------------------------------
# Frame Directory
# ------------------------------------

FRAME_DIR = "frames"

# ------------------------------------
# Get Frames
# ------------------------------------

frame_files = sorted(
    [
        file
        for file in os.listdir(
            FRAME_DIR
        )
        if file.endswith(".jpg")
        or file.endswith(".jpeg")
        or file.endswith(".png")
    ]
)

if not frame_files:

    raise FileNotFoundError(
        "No frames found. Run 31_extract_frames.py first."
    )

# ------------------------------------
# Analyze Frames
# ------------------------------------

frame_descriptions = []

print(
    "\nAnalyzing Video Frames...\n"
)

for frame in frame_files:

    frame_path = os.path.join(
        FRAME_DIR,
        frame
    )

    print(
        f"Analyzing {frame}"
    )

    description = describe_image(
        frame_path,
        prompt="""
Describe what is happening in this frame.

Focus on:
- People
- Actions
- Objects
- Scene changes
"""
    )

    frame_descriptions.append(
        f"{frame}:\n{description}"
    )

# ------------------------------------
# Print Frame Descriptions
# ------------------------------------

print("\n")
print("=" * 60)
print("FRAME DESCRIPTIONS")
print("=" * 60)

for desc in frame_descriptions:

    print("\n")
    print(desc)

# ------------------------------------
# Create Video Summary Prompt
# ------------------------------------

video_context = "\n\n".join(
    frame_descriptions
)

summary_prompt = f"""
You are a video understanding assistant.

The following descriptions come from
sequential frames extracted from a video.

{video_context}

Tasks:

1. Explain what is happening in the video.
2. Describe the sequence of events.
3. Identify important objects, people, or actions.
4. Provide a concise final summary.

Return a structured response.
"""

# ------------------------------------
# Generate Summary
# ------------------------------------

print("\n")
print("=" * 60)
print("GENERATING VIDEO SUMMARY")
print("=" * 60)

summary_response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=800,
    messages=[
        {
            "role": "user",
            "content": summary_prompt
        }
    ]
)

video_summary = (
    summary_response.content[0].text
)

# ------------------------------------
# Output Summary
# ------------------------------------

print("\n")
print("=" * 60)
print("VIDEO SUMMARY")
print("=" * 60)

print(
    video_summary
)