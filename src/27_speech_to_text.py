import whisper
import os

# ------------------------------------
# Configuration
# ------------------------------------

AUDIO_FILE = "audio/sample.mp3"

# Available models:
# tiny, base, small, medium, large

MODEL_NAME = "base"

# ------------------------------------
# Validate File
# ------------------------------------

if not os.path.exists(AUDIO_FILE):
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_FILE}"
    )

# ------------------------------------
# Load Whisper Model
# ------------------------------------

print("\nLoading Whisper model...\n")

model = whisper.load_model(
    MODEL_NAME
)

# ------------------------------------
# Transcribe Audio
# ------------------------------------

print("Transcribing audio...\n")

result = model.transcribe(
    AUDIO_FILE
)

# ------------------------------------
# Output Transcript
# ------------------------------------

print("=" * 50)
print("TRANSCRIPT")
print("=" * 50)

print(result["text"])

print("\nDone.")