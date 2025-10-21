import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path

# ------------------------------
# params
project_root = Path(__file__).resolve().parents[2]  # assumes script is under S-Sense-Project/Sight/...
input_path  = project_root / "Sight/eularian-video-magnification/data/tests-jg/throat-IN.mp4"
output_path = project_root / "Sight/eularian-video-magnification/data/tests-jg/throat-OUT.mp4"

alpha       = 50.0     # amplification factor
fl, fh      = 0.8, 2.0 # band (Hz): 0.8–2.0 ~ heart-rate
pyr_scale   = 0.5      # downsample factor for speed (0.5 = half-size)
levels      = 0        # Gaussian pyramid levels (0 = none)
input_scale = 0.5      # scale input video to 50%
# ------------------------------

# open video
cap = cv2.VideoCapture(str(input_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * input_scale)
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * input_scale)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

# read a first frame
ok, frame0 = cap.read()
if not ok:
    raise RuntimeError("Could not read first frame")

# scale before ROI
frame0 = cv2.resize(frame0, (w, h), interpolation=cv2.INTER_AREA)

disp = frame0.copy()
roi = cv2.selectROI("Select ROI (press ENTER)", disp, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()
x, y, w_roi, h_roi = map(int, roi)
if w_roi == 0 or h_roi == 0:
    raise RuntimeError("Empty ROI selected")

# build a list of scaled frames
frames = [frame0]
while True:
    ok, f = cap.read()
    if not ok:
        break
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    frames.append(f)
cap.release()

# helper functions
def downsample(img, levels, scale):
    out = img.copy()
    if levels > 0:
        for _ in range(levels):
            out = cv2.pyrDown(out)
    elif scale != 1.0:
        out = cv2.resize(out, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return out

def upsample(img, target_shape, levels, scale):
    out = img.copy()
    if levels > 0:
        for _ in range(levels):
            out = cv2.pyrUp(out)
    elif scale != 1.0:
        out = cv2.resize(out, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return out

def to_y(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)

# extract ROI time-series (in Y channel for stability)
roi_series = []
for f in frames:
    patch = f[y:y+h_roi, x:x+w_roi]
    patch_ds = downsample(patch, levels, pyr_scale)
    roi_series.append(to_y(patch_ds))
roi_series = np.stack(roi_series, axis=0)

T, Hs, Ws = roi_series.shape

# design bandpass filter
nyq = 0.5 * fps
b, a = butter(N=2, Wn=[fl/nyq, fh/nyq], btype='bandpass')

# temporal filtering
roi_series_f = roi_series.reshape(T, -1)
roi_series_bp = filtfilt(b, a, roi_series_f, axis=0)
roi_series_bp = roi_series_bp.reshape(T, Hs, Ws)

# amplify
roi_series_amp = roi_series_bp * alpha

# reconstruct and write output
for i, f in enumerate(frames):
    patch = f[y:y+h_roi, x:x+w_roi]
    patch_ds = downsample(patch, levels, pyr_scale)

    ycrcb = cv2.cvtColor(patch_ds, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_chan = ycrcb[:,:,0]

    y_mag = y_chan + roi_series_amp[i]
    y_mag = np.clip(y_mag, 0, 255)

    ycrcb[:,:,0] = y_mag
    patch_recon = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    patch_full  = upsample(patch_recon, (h_roi, w_roi), levels, pyr_scale)

    out_frame = f.copy()
    out_frame[y:y+h_roi, x:x+w_roi] = patch_full
    out.write(out_frame)

out.release()
print(f"\n✅ Saved to: {output_path}")
