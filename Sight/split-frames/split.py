import cv2
import os

# ------------------------------
# config — relative to project root (Sight/)
project_root = os.path.dirname(os.path.abspath(__file__))  # e.g. .../Sight/split-frames
sight_root = os.path.abspath(os.path.join(project_root, ".."))  # go up to Sight/

video_path = os.path.join(sight_root, "split-frames", "tests-jg3", "neck-IN.mp4")
output_folder = os.path.join(sight_root, "split-frames", "dataset", "frames")
os.makedirs(output_folder, exist_ok=True)
# ------------------------------

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open video: {video_path}")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Process ===
    # Rotate 90° clockwise → portrait
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Optionally scale down to 50%
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # Save each frame as JPG
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"✅ Extracted {frame_count} portrait frames to {output_folder}")
