import cv2
import os

VIDEO_PATH = "videos/sample.mp4"

OUTPUT_DIR = "frames"

FRAME_EVERY_SECONDS = 5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_interval = int(
    fps * FRAME_EVERY_SECONDS
)

count = 0
saved = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    if count % frame_interval == 0:

        filename = (
            f"{OUTPUT_DIR}/frame_{saved}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Saved {filename}"
        )

        saved += 1

    count += 1

cap.release()

print(
    f"\nExtracted {saved} frames"
)