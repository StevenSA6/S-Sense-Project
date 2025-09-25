import cv2
import numpy as np
import os

# ------------------- User params -------------------
input_path   = "C:\\Workspace\\ProgProj\\filters\\ss\\data\\tests\\neck-IN.mp4"   # your face video
output_base  = "C:\\Workspace\\ProgProj\\filters\\ss\\data\\tests\\neck-OUT"    # outputs: skin_out_overlay.mp4, skin_out_mask.mp4
use_roi      = True          # select a rectangle ROI on first frame
hair_suppression = True      # reduce beard/hair via gradient gate
grad_thresh  = 25            # 10–40; higher = less suppression
kernel_size  = 5             # morphology kernel size for cleaning mask
# ---------------------------------------------------

def skin_mask_hsv(bgr):
    """Broad HSV skin band (tune under different lighting/tones)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # H: 0–179, S: 0–255, V: 0–255
    lower = np.array([0,   25,  60], dtype=np.uint8)
    upper = np.array([25, 200, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower, upper)

    # Optional extra band near red hues (wrap-around). Often not needed; comment in if helpful.
    # lower2 = np.array([160, 25,  60], dtype=np.uint8)
    # upper2 = np.array([180, 200, 255], dtype=np.uint8)
    # m1 = cv2.bitwise_or(m1, cv2.inRange(hsv, lower2, upper2))
    return m1

def skin_mask_ycrcb(bgr):
    """Classic YCrCb skin band (Cr/Cb box) — more lighting-robust."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycc)
    # Common literature ranges:
    #   Cr in [133, 173], Cb in [77, 127]
    # Loosen slightly for variation
    cr_low, cr_high = 128, 180
    cb_low, cb_high = 70, 135
    m2 = cv2.inRange(Cr, cr_low, cr_high) & cv2.inRange(Cb, cb_low, cb_high)
    return m2

def suppress_hair(mask, bgr, thresh=25):
    """Knock out high-frequency hair pixels from mask using gradient magnitude."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.addWeighted(cv2.absdiff(gx, 0), 1.0, cv2.absdiff(gy, 0), 1.0, 0))
    hair = (mag > thresh).astype(np.uint8) * 255
    # remove hair from mask
    clean = cv2.bitwise_and(mask, cv2.bitwise_not(hair))
    return clean

def build_skin_mask(bgr, hair_gate=True, grad_thresh=25, ksize=5):
    """Fuse HSV+YCrCb → clean morphology → optional hair suppression."""
    m_hsv  = skin_mask_hsv(bgr)
    m_ycc  = skin_mask_ycrcb(bgr)
    m      = cv2.bitwise_and(m_hsv, m_ycc)

    # morphology to clean speckles / fill small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    if hair_gate:
        m = suppress_hair(m, bgr, thresh=grad_thresh)

    # final smooth
    m = cv2.medianBlur(m, 5)
    return m  # 0/255

def overlay_mask(bgr, mask, alpha=0.55):
    """Colorize mask and overlay on original frame."""
    heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    out  = cv2.addWeighted(bgr, 1.0, heat, alpha, 0)
    return out

def main():
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_overlay = cv2.VideoWriter(f"{output_base}_overlay.mp4", fourcc, fps, (W, H))
    out_mask    = cv2.VideoWriter(f"{output_base}_mask.mp4",    fourcc, fps, (W, H), isColor=False)

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    x = y = w = h = None
    if use_roi:
        r = cv2.selectROI("Select ROI (press ENTER)", first, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        x, y, w, h = map(int, r)
        if w == 0 or h == 0:
            x=y=w=h=None  # fall back to full frame

    # Rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if w is None:
            bgr = frame
            mask = build_skin_mask(bgr, hair_gate=hair_suppression, grad_thresh=grad_thresh, ksize=kernel_size)
            vis  = overlay_mask(bgr, mask)
        else:
            # Process only ROI, keep rest black in mask
            bgr = frame[y:y+h, x:x+w]
            roi_mask = build_skin_mask(bgr, hair_gate=hair_suppression, grad_thresh=grad_thresh, ksize=kernel_size)

            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            mask[y:y+h, x:x+w] = roi_mask

            vis  = overlay_mask(frame, mask)

        out_overlay.write(vis)
        out_mask.write(mask)

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out_overlay.release()
    out_mask.release()
    print(f"Saved: {output_base}_overlay.mp4 and {output_base}_mask.mp4")

if __name__ == "__main__":
    main()
