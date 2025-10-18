import cv2
import numpy as np
import os
from pathlib import Path

# ------------------------------
# config (relative to project root)
project_root = Path(__file__).resolve().parents[2]  
input_path   = project_root / "Sight/optical-flow/data/tests-ik/throat-IN.mp4"
output_path  = project_root / "Sight/optical-flow/data/tests-ik/throat-OUT.mp4"

scale_factor = 0.5  # shrink video to 50%
# ------------------------------

cap = cv2.VideoCapture(str(input_path))
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open input video: {input_path}")

# Swallow counter 
swallow_count = 0
swallow_active = False

# Get FPS and scaled frame size
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(orig_width * scale_factor)
height = int(orig_height * scale_factor)

# Prepare output directory 
os.makedirs(output_path.parent, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# ROI selection 
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("❌ Could not read the first frame from input video")

# scale frame before ROI
first_frame = cv2.resize(first_frame, (width, height), interpolation=cv2.INTER_AREA)
roi_box = cv2.selectROI("Draw ROI around Adam's apple", first_frame, False, False)
cv2.destroyWindow("Draw ROI around Adam's apple")

x, y, w, h = map(int, roi_box)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Processing loop 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # scale each frame before analysis
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_curr = gray[y:y+h, x:x+w]

    flow = cv2.calcOpticalFlowFarneback(roi_prev, roi_curr, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    _, motion_thresh = cv2.threshold(magnitude, 0.5, 255, cv2.THRESH_TOZERO)

    motion_vis = cv2.normalize(motion_thresh, None, 0, 255, cv2.NORM_MINMAX)
    motion_vis = cv2.GaussianBlur(motion_vis, (9, 9), 0)
    motion_vis = cv2.applyColorMap(motion_vis.astype(np.uint8), cv2.COLORMAP_JET)

    blended = cv2.addWeighted(frame[y:y+h, x:x+w], 0.5, motion_vis, 0.5, 0)
    frame[y:y+h, x:x+w] = blended

    motion_mask = (motion_thresh > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    adams_present = False
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x
                cy = int(M["m01"] / M["m00"]) + y
                adams_present = True
                cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

    if not adams_present and not swallow_active:
        swallow_count += 1
        swallow_active = True
        print(f"Swallow detected! Count = {swallow_count}")

    if adams_present:
        swallow_active = False

    color = (0, 255, 0) if adams_present else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Swallow Detector", frame)
    prev_gray = gray.copy()

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Output saved to: {output_path}")
