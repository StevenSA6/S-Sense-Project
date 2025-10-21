import cv2
import numpy as np
import os
import math
from pathlib import Path

# ------------------------------
# paths (relative to S-Sense-Project)
project_root = Path(__file__).resolve().parents[2]  # → .../S-Sense-Project
input_path   = project_root / "Sight/gabor/data/tests/face-IN.mp4"
output_root  = project_root / "Sight/gabor/data/tests/face-out"

use_roi_for_video = True   # whether to select ROI
input_scale = 0.5          # scale input video to 50%
# ------------------------------

# Gabor bank params
ksize   = 31
sigmas  = [4.0]
gammas  = [0.5]
psis    = [0]
lambdas = [6, 8, 12]
n_orients = 8
ema_alpha = 0.3
vis_gain = 2.0
# ------------------------------


def build_gabor_kernels(ksize, sigmas, gammas, psis, lambdas, n_orients):
    kernels = []
    thetas = [i * (np.pi / n_orients) for i in range(n_orients)]
    for theta in thetas:
        for sigma in sigmas:
            for gamma in gammas:
                for psi in psis:
                    for lam in lambdas:
                        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, psi, ktype=cv2.CV_32F)
                        kern -= kern.mean()
                        s = np.sum(np.abs(kern))
                        if s > 1e-8:
                            kern /= s
                        kernels.append((theta, sigma, gamma, psi, lam, kern))
    return kernels


def apply_gabor_bank(gray, kernels):
    h, w = gray.shape
    energy = np.zeros((h, w), np.float32)
    responses = []

    for meta in kernels:
        theta, sigma, gamma, psi, lam, kern = meta
        resp = cv2.filter2D(gray, cv2.CV_32F, kern)
        mag = np.abs(resp)
        responses.append((meta, mag))
        energy += mag

    return responses, energy


def overlay_heatmap(bgr, scalar_map, gain=1.0, alpha=0.6):
    m = scalar_map.copy()
    m = m * gain
    m -= m.min()
    vmax = m.max()
    if vmax > 1e-6:
        m = m / vmax
    heat = (m * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(bgr, 1.0, heat_color, alpha, 0)
    return out


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_video(video_path, out_basename, kernels, use_roi=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(W_in * input_scale)
    H = int(H_in * input_scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(f"{out_basename}.mp4")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame")

    # Scale the first frame before ROI
    first_scaled = cv2.resize(first, (W, H), interpolation=cv2.INTER_AREA)

    x = y = w = h = None
    if use_roi:
        roi = cv2.selectROI("Select ROI (press ENTER)", first_scaled, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        x, y, w, h = map(int, roi)
        if w == 0 or h == 0:
            x = y = w = h = None

    ema_energy = None
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Scale every frame before analysis
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if w is not None:
            roi_gray = gray_full[y:y+h, x:x+w]
            _, energy = apply_gabor_bank(roi_gray, kernels)
            energy_full = np.zeros_like(gray_full, dtype=np.float32)
            energy_full[y:y+h, x:x+w] = energy
        else:
            _, energy_full = apply_gabor_bank(gray_full, kernels)

        # Smooth temporal energy map
        if ema_energy is None:
            ema_energy = energy_full
        else:
            ema_energy = (1.0 - ema_alpha) * ema_energy + ema_alpha * energy_full

        overlay = overlay_heatmap(frame, ema_energy, gain=vis_gain, alpha=0.6)
        writer.write(overlay)

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"[Video] Saved {out_path}")


def main():
    kernels = build_gabor_kernels(
        ksize=ksize,
        sigmas=sigmas,
        gammas=gammas,
        psis=psis,
        lambdas=lambdas,
        n_orients=n_orients
    )

    ext = input_path.suffix.lower()
    if ext in [".mp4", ".avi", ".mov", ".mkv", ".m4v"]:
        base = output_root.with_suffix("")
        process_video(input_path, base, kernels, use_roi=use_roi_for_video)
    else:
        raise ValueError("Unsupported input type. Use a video file.")


if __name__ == "__main__":
    main()
