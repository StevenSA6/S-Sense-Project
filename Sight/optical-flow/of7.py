import cv2
import numpy as np

# ---------- Config ----------
input_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\optical-flow\data\tests-jg2\neck-IN.mp4"

# Motion History Image params
MHI_MAX = 255.0
MHI_TAU_FRAMES = 30
MHI_DECAY = MHI_MAX / MHI_TAU_FRAMES

# Event detection params
EMA_ALPHA = 0.3
BASELINE_ALPHA = 0.02
HIGH_THRESH = 0.12
LOW_THRESH  = 0.06
REFRACTORY_SEC = 1.2

# Preproc
DIFF_THRESH = 18
BLUR_K = 5
MIN_BLOB_AREA = 300

# Display / Orientation
DISPLAY_SCALE = 0.5  # 50% size for display windows
PREFER_ROTATE_FLAG = cv2.ROTATE_90_CLOCKWISE  # use ROTATE_90_COUNTERCLOCKWISE if upside down

# ----------------------------

def rotate_to_portrait(frame, prefer_flag=PREFER_ROTATE_FLAG):
    """
    Ensures the frame is portrait (height > width).
    If the frame is landscape (width >= height), rotate by prefer_flag.
    Otherwise, return as-is.
    """
    h, w = frame.shape[:2]
    if w >= h:
        return cv2.rotate(frame, prefer_flag)
    return frame

def resize_for_display(frame, scale=DISPLAY_SCALE):
    """Resize frame for display (default 50%)."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
delay = int(1000 / fps)
refractory_frames = max(1, int(REFRACTORY_SEC * fps))

# ---- First frame & ROI selection ----
ret, first_frame_raw = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

# Force portrait orientation
first_frame = rotate_to_portrait(first_frame_raw)

# Show first frame scaled to 50% for ROI selection
roi_window = "Draw ROI around Adam's apple / throat"
cv2.namedWindow(roi_window, cv2.WINDOW_NORMAL)
first_frame_disp = resize_for_display(first_frame)

roi_scaled = cv2.selectROI(roi_window, first_frame_disp, showCrosshair=False, fromCenter=False)
cv2.destroyWindow(roi_window)

if roi_scaled is None or roi_scaled == (0, 0, 0, 0):
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit("ROI selection cancelled or invalid.")

# Map ROI back to full-resolution rotated frame coordinates
sx, sy, sw, sh = map(int, roi_scaled)
scale = DISPLAY_SCALE
x = int(round(sx / scale))
y = int(round(sy / scale))
w = int(round(sw / scale))
h = int(round(sh / scale))

# Clamp ROI to frame bounds
H, W = first_frame.shape[:2]
x = max(0, min(x, W - 1))
y = max(0, min(y, H - 1))
w = max(1, min(w, W - x))
h = max(1, min(h, H - y))

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Motion History Image buffer
mhi = np.zeros((h, w), dtype=np.float32)

# Event state
swallow_count = 0
frames_since_last = 9999
centroid_ema = None
baseline = None
swallow_active = False

# Create resizable main display window
main_win = "Swallow Detector (with motion tracking)"
cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)

while True:
    ret, frame_raw = cap.read()
    if not ret:
        break

    # Rotate each frame to portrait
    frame = rotate_to_portrait(frame_raw)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- ROI crops on rotated frames ---
    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_curr = gray[y:y+h, x:x+w]

    # --- Motion mask from frame diff ---
    if BLUR_K > 1:
        roi_prev_blur = cv2.GaussianBlur(roi_prev, (BLUR_K, BLUR_K), 0)
        roi_curr_blur = cv2.GaussianBlur(roi_curr, (BLUR_K, BLUR_K), 0)
    else:
        roi_prev_blur, roi_curr_blur = roi_prev, roi_curr

    diff = cv2.absdiff(roi_prev_blur, roi_curr_blur)
    _, motion_bin = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # --- Update MHI ---
    mhi = np.maximum(0.0, mhi - MHI_DECAY)
    mhi[motion_bin > 0] = MHI_MAX

    # --- Compute vertical centroid of motion ---
    delta_norm = 0.0
    if cv2.countNonZero((mhi > (0.3 * MHI_MAX)).astype(np.uint8)) > MIN_BLOB_AREA:
        weights = mhi / (MHI_MAX + 1e-6)
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)
        col_weights = np.sum(weights, axis=1)
        total_w = np.sum(col_weights) + 1e-6
        centroid_y = float(np.sum(y_coords[:, 0] * col_weights) / total_w)

        # Smooth centroid
        if centroid_ema is None:
            centroid_ema = centroid_y
        else:
            centroid_ema = EMA_ALPHA * centroid_y + (1 - EMA_ALPHA) * centroid_ema

        # Maintain baseline (when not in swallow)
        if baseline is None:
            baseline = centroid_ema
        else:
            if not swallow_active:
                baseline = BASELINE_ALPHA * centroid_ema + (1 - BASELINE_ALPHA) * baseline

        # Normalized upward deflection (up = smaller y)
        delta_norm = (baseline - centroid_ema) / max(1.0, float(h))

        # Hysteresis state machine
        if not swallow_active:
            if delta_norm > HIGH_THRESH and frames_since_last > refractory_frames:
                swallow_count += 1
                swallow_active = True
                frames_since_last = 0
                print(f"Swallow detected! Count = {swallow_count}")
        else:
            if delta_norm < LOW_THRESH:
                swallow_active = False

    frames_since_last += 1

    # ---------- Visualization ----------
    mhi_vis = np.clip(mhi, 0, 255).astype(np.uint8)
    mhi_vis = cv2.GaussianBlur(mhi_vis, (9, 9), 0)
    mhi_vis_c = cv2.applyColorMap(mhi_vis, cv2.COLORMAP_JET)

    roi_color = frame[y:y+h, x:x+w]
    frame[y:y+h, x:x+w] = cv2.addWeighted(roi_color, 0.5, mhi_vis_c, 0.5, 0)

    # ROI box color: green idle, red if active
    box_color = (0, 255, 0) if not swallow_active else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

    # Draw centroid and baseline lines
    if centroid_ema is not None:
        cy_abs = y + int(round(centroid_ema))
        cv2.line(frame, (x, cy_abs), (x+w, cy_abs), (255, 255, 255), 1)  # centroid
    if baseline is not None:
        by_abs = y + int(round(baseline))
        cv2.line(frame, (x, by_abs), (x+w, by_abs), (0, 255, 255), 1)    # baseline

    # HUD
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 240, 40), 2)
    cv2.putText(frame, f"delta: {delta_norm:+.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 0), 2)

    # Show at 50% in a resizable window
    frame_disp = resize_for_display(frame)
    cv2.imshow(main_win, frame_disp)

    prev_gray = gray

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
