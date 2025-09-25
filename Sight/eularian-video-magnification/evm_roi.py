import cv2
import numpy as np
from scipy.signal import butter, filtfilt

# --- params ---
input_path  = "C:\\Workspace\\ProgProj\\filters\\eularian-video-magnification\\data\\tests-ik\\face-IN.mp4"
output_path = "C:\\Workspace\\ProgProj\\filters\\eularian-video-magnification\\data\\tests-ik\\face-OUT.mp4"
alpha       = 50.0           # amplification factor
fl, fh      = 0.8, 2.0       # band (Hz): 0.8–2.0 ~ heart-rate
pyr_scale   = 0.5            # downsample factor for speed (0.5 = half-size)
levels      = 0              # set >0 to use a simple Gaussian pyramid (0 means none)
# ------------------------------

# open video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Could not open input video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# read a first frame to pick ROI
ok, frame0 = cap.read()
if not ok:
    raise RuntimeError("Could not read first frame")

disp = frame0.copy()
roi = cv2.selectROI("Select ROI (press ENTER)", disp, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()
x, y, w_roi, h_roi = map(int, roi)
if w_roi == 0 or h_roi == 0:
    raise RuntimeError("Empty ROI selected")

# build a list of frames for one pass processing 
frames = [frame0]
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()

# optional spatial downsample for speed/noise
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

# extract ROI time-series (in Y channel for stability)
def to_y(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)

# stack ROI over time (after spatial downsample)
roi_series = []
for f in frames:
    patch = f[y:y+h_roi, x:x+w_roi]
    patch_ds = downsample(patch, levels, pyr_scale)
    roi_series.append(to_y(patch_ds))
roi_series = np.stack(roi_series, axis=0)  # shape: (T, H', W')

T, Hs, Ws = roi_series.shape

# design bandpass
nyq = 0.5 * fps
b, a = butter(N=2, Wn=[fl/nyq, fh/nyq], btype='bandpass')

# filter each pixel’s time-series
roi_series_f = roi_series.reshape(T, -1)
roi_series_bp = filtfilt(b, a, roi_series_f, axis=0)
roi_series_bp = roi_series_bp.reshape(T, Hs, Ws)

# amplify
roi_series_amp = roi_series_bp * alpha

# add back to original luminance in ROI and rebuild frames
for i, f in enumerate(frames):
    patch = f[y:y+h_roi, x:x+w_roi]
    patch_ds = downsample(patch, levels, pyr_scale)

    # get YCrCb
    ycrcb = cv2.cvtColor(patch_ds, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_chan = ycrcb[:,:,0]

    y_mag = y_chan + roi_series_amp[i]
    y_mag = np.clip(y_mag, 0, 255)

    ycrcb[:,:,0] = y_mag
    patch_recon = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    patch_full  = upsample(patch_recon, (h_roi, w_roi), levels, pyr_scale)

    # composite into original frame
    out_frame = f.copy()
    out_frame[y:y+h_roi, x:x+w_roi] = patch_full
    out.write(out_frame)

out.release()
print("Saved:", output_path)
