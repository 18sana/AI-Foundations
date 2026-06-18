# pyrefly: ignore [missing-import]
import pyttsx3

# -------------------------
# Initialize Engine
# -------------------------

engine = pyttsx3.init()

# -------------------------
# Text Input
# -------------------------

text = input(
    "\nEnter text: "
)

# -------------------------
# Speak Text
# -------------------------

engine.say(text)

engine.runAndWait()

print(
    "\nSpeech completed."
)