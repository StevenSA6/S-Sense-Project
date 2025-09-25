import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import csv, os

# ---------------- User params ----------------
input_path   = "C:\\Workspace\\ProgProj\\filters\\optical-flow\\data\\tests-jg2\\neck-IN.mp4"   
output_base  = "C:\\Workspace\\ProgProj\\filters\\optical-flow\\data\\tests-jg2\\neck-OUT"      
use_roi      = True             # select throat ROI on first frame
metric_type  = "mag"            # | "v" | "div" | "mag+v"
band_hz      = (0.3, 2.0)       # passband (Hz): ~0.3–2.0 works for swallow bursts
z_threshold  = 3.0              # trigger when z-score crosses this
refractory_s = 1.5              # seconds to suppress after a trigger
gauss_sigma  = 0.8              # blur gray to reduce noise; set 0 to disable
draw_flow_quiver = False        # overlay sparse flow arrows (slow if True)
# ---------------------------------------------

def bandpass_z(series, fs, band):
    series = np.asarray(series, np.float32)
    series = series - np.nanmean(series)
    if len(series) < 20:
        # too short for filtfilt—fallback to simple standardization (requires more research to understand param)
        z = (series - np.mean(series)) / (np.std(series) + 1e-6)
        return z
    nyq = 0.5 * fs
    lo, hi = max(1e-3, band[0]/nyq), min(0.999, band[1]/nyq)
    b, a = butter(N=2, Wn=[lo, hi], btype='bandpass')
    bp = filtfilt(b, a, series, method="gust")
    z  = (bp - np.mean(bp)) / (np.std(bp) + 1e-6)
    return z

def detect_events(z, thr, refractory_frames):
    events = []
    armed = True
    last_fire = -1e9
    for i, val in enumerate(z):
        if armed and val >= thr:
            events.append(i)
            last_fire = i
            armed = False
        if not armed and i - last_fire >= refractory_frames:
            armed = True
    return events

def compute_metrics(prev_gray, gray):
    # Farneback dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=24,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    u = flow[...,0]; v = flow[...,1]
    mag = np.hypot(u, v)

    if metric_type == "mag":
        m = np.mean(mag)
    elif metric_type == "v":
        m = float(np.mean(v))       # vertical component motion
    elif metric_type == "div":
        du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
        dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
        div   = du_dx + dv_dy
        m = float(np.mean(div))
    elif metric_type == "mag+v":
        m = float(np.mean(mag) + 0.5*np.mean(np.abs(v)))
    else:
        m = np.mean(mag)

    return m, flow

def draw_quiver(img, flow, step=16):
    h, w = flow.shape[:2]
    vis = img.copy()
    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            fx, fy = flow[y, x]
            end = (int(x+fx), int(y+fy))
            cv2.arrowedLine(vis, (x,y), end, (0,255,255), 1, tipLength=0.3)
    return vis

def main():
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(f"{output_base}_overlay.mp4", fourcc, fps, (W, H))

    # First frame & ROI
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    if use_roi:
        r = cv2.selectROI("Select throat ROI (press ENTER)", first, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        x, y, w, h = map(int, r)
        if w == 0 or h == 0:
            x=y=0; w=W; h=H
    else:
        x=y=0; w=W; h=H

    # Rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_gray = None
    metrics = []
    frames = 0

    # pass 1: compute metrics & collect frames for overlay timestamps
    all_rois = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gauss_sigma and gauss_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0,0), gauss_sigma)
        if prev_gray is None:
            prev_gray = gray
            metrics.append(0.0)   # first frame: no flow
            all_rois.append(roi)
            frames += 1
            continue

        m, flow = compute_metrics(prev_gray, gray)
        metrics.append(float(m))
        all_rois.append(roi)
        prev_gray = gray
        frames += 1

    cap.release()

    # process metric series → z-score
    z = bandpass_z(metrics, fs=fps, band=band_hz)
    refractory_frames = int(round(refractory_s * fps))
    event_idxs = detect_events(z, z_threshold, refractory_frames)

    # pass 2: write overlay video and CSV
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to reopen {input_path}")

    # Prepare CSV
    csv_path = f"{output_base}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "metric_raw", "z_score", "trigger"])
        i = -1
        prev_gray = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            t = i / fps
            trig = 1 if i in event_idxs else 0
            raw = metrics[i] if i < len(metrics) else 0.0
            zi  = float(z[i]) if i < len(z) else 0.0
            writer.writerow([i, f"{t:.3f}", f"{raw:.6f}", f"{zi:.3f}", trig])

            # Draw ROI & text overlays
            vis = frame.copy()
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(vis, f"metric={raw:.4f}  z={zi:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if trig:
                cv2.putText(vis, "SWALLOW", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                cv2.circle(vis, (x+w//2, y+h//2), 20, (0,0,255), 3)

            out_vid.write(vis)

    cap.release()
    out_vid.release()

    # Print event times
    if event_idxs:
        times = [idx / fps for idx in event_idxs]
        print("Detected swallow events (s):", ", ".join(f"{t:.2f}" for t in times))
    else:
        print("No events detected. Try lowering z_threshold or adjusting band_hz.")

    print(f"Saved: {output_base}_overlay.mp4 and {csv_path}")

if __name__ == "__main__":
    main()
