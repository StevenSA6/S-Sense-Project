import cv2
import numpy as np

# Input path
input_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\optical-flow\data\tests-jg2\face-IN.mp4"
cap = cv2.VideoCapture(input_path)

# Swallow counter
swallow_count = 0
swallow_detected = False
frames_since_last = 999  # frames since last swallow

# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30
min_swallow_gap = int(fps * 1.5)  # require at least 1.5 sec between swallows

# === Step 1: Let user click to place line center ===
line_y = None
line_x = None
line_length = 200  # pixels wide, adjust to suit

def set_line(event, x, y, flags, param):
    global line_y, line_x
    if event == cv2.EVENT_LBUTTONDOWN:
        line_y, line_x = y, x
        print(f"Line set at (x={line_x}, y={line_y})")

ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    cap.release()
    exit()

clone = first_frame.copy()
cv2.namedWindow("Set Line (click once)")
cv2.setMouseCallback("Set Line (click once)", set_line)

while line_y is None:
    disp = clone.copy()
    cv2.putText(disp, "Click to set line across neck", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Set Line (click once)", disp)
    if cv2.waitKey(20) & 0xFF == 27:  # ESC quits
        break

cv2.destroyWindow("Set Line (click once)")

# Convert first frame to gray for motion comparison
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Step 2: Motion detection in strip around line ===
    strip_h = 60  # band height around line
    y1, y2 = max(0, line_y - strip_h//2), min(gray.shape[0], line_y + strip_h//2)

    half_len = line_length // 2
    x1, x2 = max(0, line_x - half_len), min(gray.shape[1], line_x + half_len)

    strip_prev = prev_gray[y1:y2, x1:x2]
    strip_curr = gray[y1:y2, x1:x2]

    # Frame difference for motion
    diff = cv2.absdiff(strip_prev, strip_curr)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Largest contour = strongest motion blob
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    adams_y = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:  # ignore small blobs
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                adams_y = cy
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

    # === Step 3: Check line crossing with debounce ===
    if adams_y is not None:
        if adams_y < line_y and not swallow_detected and frames_since_last > min_swallow_gap:
            swallow_count += 1
            swallow_detected = True
            frames_since_last = 0
            print(f"Swallow detected! Count = {swallow_count}")

        if adams_y >= line_y:
            swallow_detected = False

    # Draw short line across neck
    cv2.line(frame, (x1, line_y), (x2, line_y), (0, 255, 0), 2)

    # Counter overlay
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Swallow Detector", frame)
    prev_gray = gray.copy()
    frames_since_last += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
