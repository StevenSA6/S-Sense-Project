import cv2
import os

# --- Configuration ---
video_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\split-frames\tests-jg3\throat-IN.mp4"
output_folder = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\split-frames\dataset\frames"
os.makedirs(output_folder, exist_ok=True)

# --- Open video ---
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save each frame as JPG
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"✅ Extracted {frame_count} frames to {output_folder}")
