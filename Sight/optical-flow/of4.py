import cv2
import numpy as np

# =========================
# Config / Input
# =========================
input_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\optical-flow\data\tests-jg2\face-IN.mp4"
cap = cv2.VideoCapture(input_path)

# Choose rotation so video is vertical on screen:
#   cv2.ROTATE_90_CLOCKWISE or cv2.ROTATE_90_COUNTERCLOCKWISE or cv2.ROTATE_180
ROTATE_FLAG = cv2.ROTATE_90_CLOCKWISE  # change if upside-down

# Swallow counter / timing
swallow_count = 0
swallow_detected = False
frames_since_last = 999  # frames since last swallow

# FPS / delays
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps and fps > 0 else 30
min_swallow_gap = int((fps if fps and fps > 0 else 30) * 1.5)  # require ~1.5 sec between swallows

# Line placement
line_y = None
line_x = None
line_length = 200  # pixels wide, adjust to suit

# Default scaling factor (50%)
DISPLAY_SCALE = 0.5

def rotate_if_needed(frame):
    if ROTATE_FLAG is not None:
        return cv2.rotate(frame, ROTATE_FLAG)
    return frame

def resize_for_display(frame, scale=DISPLAY_SCALE):
    """Resize frame to a smaller display size while keeping aspect ratio."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

def set_line(event, x, y, flags, param):
    global line_y, line_x
    if event == cv2.EVENT_LBUTTONDOWN:
        # scale back click coords to full-size frame
        line_y, line_x = int(y / DISPLAY_SCALE), int(x / DISPLAY_SCALE)
        print(f"Line set at (x={line_x}, y={line_y})")

# =========================
# Read & prepare first frame
# =========================
ret, first_frame = cap.read()
if not ret or first_frame is None:
    print("Error: Cannot read video")
    if cap:
        cap.release()
    raise SystemExit(1)

# Rotate first frame so UI matches processing orientation
first_frame = rotate_if_needed(first_frame)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# =========================
# Ask user to click the line
# =========================
clone = first_frame.copy()
cv2.namedWindow("Set Line (click once)", cv2.WINDOW_NORMAL)  # resizable
cv2.setMouseCallback("Set Line (click once)", set_line)

while True:
    disp = resize_for_display(clone)
    cv2.putText(disp, "Click to set line across neck (ESC to cancel)", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Set Line (click once)", disp)
    k = cv2.waitKey(20) & 0xFF
    if line_y is not None:
        break
    if k == 27:  # ESC
        print("Canceled by user before setting line.")
        cv2.destroyAllWindows()
        cap.release()
        raise SystemExit(0)

cv2.destroyWindow("Set Line (click once)")

# =========================
# Main processing loop
# =========================
cv2.namedWindow("Swallow Detector", cv2.WINDOW_NORMAL)  # resizable

strip_h = 60  # band height around the line
area_min = 500  # ignore small blobs

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Rotate current frame to keep vertical display
    frame = rotate_if_needed(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Clamp ROI around the chosen line & length
    h, w = gray.shape[:2]
    y1 = max(0, line_y - strip_h // 2)
    y2 = min(h, line_y + strip_h // 2)

    half_len = line_length // 2
    x1 = max(0, line_x - half_len)
    x2 = min(w, line_x + half_len)

    if x1 >= x2 or y1 >= y2:
        x1 = max(0, min(line_x, w - 2))
        x2 = min(w, max(line_x + 1, x1 + 2))
        y1 = max(0, min(line_y, h - 2))
        y2 = min(h, max(line_y + 1, y1 + 2))

    strip_prev = prev_gray[y1:y2, x1:x2]
    strip_curr = gray[y1:y2, x1:x2]

    # Frame difference for motion
    diff = cv2.absdiff(strip_prev, strip_curr)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    adams_y = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > area_min:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                adams_y = cy
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

    # Swallow detection with debounce
    if adams_y is not None:
        if adams_y < line_y and not swallow_detected and frames_since_last > min_swallow_gap:
            swallow_count += 1
            swallow_detected = True
            frames_since_last = 0
            print(f"Swallow detected! Count = {swallow_count}")
        if adams_y >= line_y:
            swallow_detected = False

    # Draw the horizontal line
    cv2.line(frame, (x1, line_y), (x2, line_y), (0, 255, 0), 2)

    # HUD
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show scaled-down frame (50%) but still resizable
    frame_disp = resize_for_display(frame)
    cv2.imshow("Swallow Detector", frame_disp)

    # Prepare for next iteration
    prev_gray = gray
    frames_since_last += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
