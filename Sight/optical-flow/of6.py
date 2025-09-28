import cv2
import numpy as np

# ---------- Config ----------
input_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\optical-flow\data\tests-jg2\neck-IN.mp4"

# Motion History Image params
MHI_MAX = 255.0               # max intensity for fresh motion
MHI_TAU_FRAMES = 30           # how long motion lingers (frames)
MHI_DECAY = MHI_MAX / MHI_TAU_FRAMES

# Event detection params
EMA_ALPHA = 0.3               # smoothing for centroid (0..1, higher = snappier)
BASELINE_ALPHA = 0.02         # slow drift baseline of centroid
HIGH_THRESH = 0.12            # normalized upward deflection to trigger (fraction of ROI height)
LOW_THRESH  = 0.06            # drop below this to “reset” (hysteresis)
MIN_ABOVE_SEC = 0.15          # must stay above high for at least this duration (sec)
REFRACTORY_SEC = 1.2          # min time between swallows (sec)

# Preproc
DIFF_THRESH = 18              # binary threshold on frame diff
BLUR_K = 5                    # pre-blur kernel size (odd)
MIN_BLOB_AREA = 300           # ignore tiny motion blobs

# ----------------------------

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
delay = int(1000 / fps)
min_above_frames = max(1, int(MIN_ABOVE_SEC * fps))
refractory_frames = max(1, int(REFRACTORY_SEC * fps))

# ---- Select initial ROI ----
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

init_box = cv2.selectROI("Draw ROI around Adam's apple / throat", first_frame, False, False)
cv2.destroyWindow("Draw ROI around Adam's apple / throat")
x, y, w, h = map(int, init_box)

# ---- Create a CSRT tracker for ROI drift ----
tracker = None
# Try legacy then modern namespaces (OpenCV versions differ)
if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
    tracker = cv2.legacy.TrackerCSRT_create()
elif hasattr(cv2, "TrackerCSRT_create"):
    tracker = cv2.TrackerCSRT_create()
else:
    # Fallback: MIL if CSRT not available
    tracker = cv2.TrackerMIL_create()

tracker.init(first_frame, (x, y, w, h))

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# MHI buffer for ROI
mhi = np.zeros((h, w), dtype=np.float32)

# Event state
swallow_count = 0
frames_since_last = 9999
centroid_ema = None
baseline = None
above_counter = 0
swallow_active = False  # "we are in an elevated state" flag

def clip_bbox(b, W, H):
    x, y, w, h = b
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

while True:
    ret, frame = cap.read()
    if not ret:
        break

    Hf, Wf = frame.shape[:2]
    ok, box = tracker.update(frame)
    if ok:
        x, y, w, h = clip_bbox(box, Wf, Hf)
    else:
        # If tracking fails, keep last ROI (or you could re-select)
        x, y, w, h = clip_bbox((x, y, w, h), Wf, Hf)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- ROI crops ---
    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_curr = gray[y:y+h, x:x+w]

    # --- Basic motion mask from frame diff ---
    if BLUR_K > 1:
        roi_prev_blur = cv2.GaussianBlur(roi_prev, (BLUR_K, BLUR_K), 0)
        roi_curr_blur = cv2.GaussianBlur(roi_curr, (BLUR_K, BLUR_K), 0)
    else:
        roi_prev_blur, roi_curr_blur = roi_prev, roi_curr

    diff = cv2.absdiff(roi_prev_blur, roi_curr_blur)
    _, motion_bin = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # Morph open to clean noise
    motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # --- Update MHI (decay + refresh where motion occurs) ---
    # Decay
    mhi = np.maximum(0.0, mhi - MHI_DECAY)
    # Refresh
    mhi[motion_bin > 0] = MHI_MAX

    # --- Compute vertical centroid of *recent* motion ---
    adams_present = False
    centroid_y = None
    area = cv2.countNonZero((mhi > (0.3 * MHI_MAX)).astype(np.uint8))  # focus on recent motion
    if area > MIN_BLOB_AREA:
        # Weighted centroid along vertical axis
        weights = mhi / (MHI_MAX + 1e-6)
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)
        # Sum over x to get vertical distribution
        col_weights = np.sum(weights, axis=1)  # shape (h,)
        total_w = np.sum(col_weights) + 1e-6
        centroid_y = float(np.sum(y_coords[:,0] * col_weights) / total_w)
        adams_present = True

    # --- Smooth centroid & maintain baseline ---
    if adams_present:
        if centroid_ema is None:
            centroid_ema = centroid_y
        else:
            centroid_ema = EMA_ALPHA * centroid_y + (1 - EMA_ALPHA) * centroid_ema

        # Update baseline only when not currently elevated (prevents drift during a swallow)
        if baseline is None:
            baseline = centroid_ema
        else:
            if not swallow_active:
                baseline = BASELINE_ALPHA * centroid_ema + (1 - BASELINE_ALPHA) * baseline

        # Normalized upward deflection (positive when centroid moves up = smaller y)
        delta_norm = ((baseline - centroid_ema) / max(1.0, float(h)))

        # Hysteresis + min duration + refractory
        if delta_norm > HIGH_THRESH and frames_since_last > refractory_frames:
            above_counter += 1
            if above_counter >= min_above_frames and not swallow_active:
                swallow_count += 1
                swallow_active = True
                frames_since_last = 0
                print(f"Swallow detected! Count = {swallow_count}")
        else:
            # Drop out of elevated state when below low threshold
            if swallow_active and delta_norm < LOW_THRESH:
                swallow_active = False
            if delta_norm <= HIGH_THRESH:
                above_counter = 0
    else:
        # No good motion seen: slowly relax elevated state and let MHI decay handle gaps
        above_counter = 0
        # do not update baseline here to avoid drifting on blanks

    frames_since_last += 1

    # ---------- Visualization ----------
    # Heatmap from MHI (already smoothish), upscale to ROI and overlay
    mhi_vis = (np.clip(mhi, 0, 255)).astype(np.uint8)
    mhi_vis = cv2.GaussianBlur(mhi_vis, (9,9), 0)
    mhi_vis_c = cv2.applyColorMap(mhi_vis, cv2.COLORMAP_JET)
    roi_color = frame[y:y+h, x:x+w]
    frame[y:y+h, x:x+w] = cv2.addWeighted(roi_color, 0.5, mhi_vis_c, 0.5, 0)

    # ROI box color: green = baseline/idle, red = elevated (swallow phase)
    box_color = (0, 255, 0) if not swallow_active else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

    # Draw centroid & baseline lines (if available)
    if centroid_ema is not None:
        cy_abs = y + int(round(centroid_ema))
        cv2.line(frame, (x, cy_abs), (x+w, cy_abs), (255, 255, 255), 1)
    if baseline is not None:
        by_abs = y + int(round(baseline))
        cv2.line(frame, (x, by_abs), (x+w, by_abs), (0, 255, 255), 1)

    # Text overlays
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 240, 40), 2)
    if centroid_ema is not None and baseline is not None:
        delta_show = (baseline - centroid_ema) / max(1.0, float(h))
        cv2.putText(frame, f"delta: {delta_show:+.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 0), 2)

    cv2.imshow("Swallow Detector (MHI + centroid + tracker)", frame)
    prev_gray = gray

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
