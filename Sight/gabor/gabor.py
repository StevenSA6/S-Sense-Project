import cv2
import numpy as np
import os
import math

# ------------- USER PARAMS -------------
input_path   = "C:\\Workspace\\ProgProj\\filters\\gabor\\data\\tests\\face-IN.mp4"   # can be "input.jpg" or "input.mp4"
output_root  = "C:\\Workspace\\ProgProj\\filters\\gabor\\data\\tests\\face-out"  # folder for images / basename for video
use_roi_for_video = True       # when input is a video, ask for ROI
# Gabor bank params
ksize   = 31        # kernel size (odd). Larger -> more selective, slower
sigmas  = [4.0]     # Gaussian sigma(s)
gammas  = [0.5]     # spatial aspect ratios (y/x)
psis    = [0]       # phase offsets (0 or pi/2 are common)
lambdas = [6, 8, 12]  # wavelengths (pixels). Smaller -> higher frequency
n_orients = 8       # number of orientations over [0, pi)
# Video smoothing
ema_alpha = 0.3     # exponential moving average smoothing for energy map (0..1)
# Colormap intensity scale for display
vis_gain = 2.0      # multiply energy before colormap; tweak for visibility
# --------------------------------------


def build_gabor_kernels(ksize, sigmas, gammas, psis, lambdas, n_orients):
    """Return list of (theta, sigma, gamma, psi, lam, kernel)"""
    kernels = []
    thetas = [i * (np.pi / n_orients) for i in range(n_orients)]
    for theta in thetas:
        for sigma in sigmas:
            for gamma in gammas:
                for psi in psis:
                    for lam in lambdas:
                        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, psi, ktype=cv2.CV_32F)
                        # normalize to zero-mean, unit L1 for stability
                        kern -= kern.mean()
                        s = np.sum(np.abs(kern))
                        if s > 1e-8:
                            kern /= s
                        kernels.append((theta, sigma, gamma, psi, lam, kern))
    return kernels


def apply_gabor_bank(gray, kernels):
    """
    Apply all kernels; return:
      - responses: list of (meta, magnitude_response)
      - energy: combined energy map (sum of magnitudes)
    """
    h, w = gray.shape
    energy = np.zeros((h, w), np.float32)
    responses = []

    for meta in kernels:
        theta, sigma, gamma, psi, lam, kern = meta
        resp = cv2.filter2D(gray, cv2.CV_32F, kern)
        mag = np.abs(resp)  # magnitude response
        responses.append((meta, mag))
        energy += mag

    return responses, energy


def make_montage(responses, max_cols=4):
    """
    Build a quick montage of per-orientation responses (first N filters across orientations).
    We pick one wavelength per orientation (the smallest) for a compact view.
    """
    # choose one response per orientation (lowest lambda)
    per_theta = {}
    for meta, mag in responses:
        theta, sigma, gamma, psi, lam, _ = meta
        key = round(theta, 5)
        if key not in per_theta or lam < per_theta[key][0]:
            per_theta[key] = (lam, mag)

    tiles = [per_theta[k][1] for k in sorted(per_theta.keys())]
    # normalize each tile to 0-255 for view
    tiles_u8 = []
    for t in tiles:
        t_norm = t - t.min()
        if t_norm.max() > 1e-6:
            t_norm = t_norm / (t_norm.max() + 1e-8)
        t_u8 = (t_norm * 255.0).astype(np.uint8)
        tiles_u8.append(t_u8)

    if not tiles_u8:
        return None

    h, w = tiles_u8[0].shape
    cols = min(max_cols, len(tiles_u8))
    rows = math.ceil(len(tiles_u8) / cols)

    canvas = np.full((rows * h, cols * w), 0, np.uint8)
    for idx, im in enumerate(tiles_u8):
        r = idx // cols
        c = idx % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im

    return canvas


def overlay_heatmap(bgr, scalar_map, gain=1.0, alpha=0.6):
    """
    Overlay heatmap of scalar_map onto bgr.
    scalar_map should be float32; we'll normalize to 0-255 after gain.
    """
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


def process_image(img_path, out_dir, kernels):
    ensure_dir(out_dir)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    responses, energy = apply_gabor_bank(gray, kernels)
    montage = make_montage(responses)

    # save energy map + montage
    energy_vis = overlay_heatmap(img, energy, gain=vis_gain, alpha=0.65)
    cv2.imwrite(os.path.join(out_dir, "energy_overlay.jpg"), energy_vis)

    # Also save raw energy (normalized)
    e = energy - energy.min()
    if e.max() > 0:
        e = e / e.max()
    cv2.imwrite(os.path.join(out_dir, "energy_gray.jpg"), (e * 255).astype(np.uint8))

    if montage is not None:
        cv2.imwrite(os.path.join(out_dir, "responses_montage.jpg"), montage)

    print(f"[Image] Saved energy_overlay.jpg, energy_gray.jpg, responses_montage.jpg in {out_dir}")


def process_video(video_path, out_basename, kernels, use_roi=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = out_basename + ".mp4"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame")

    x = y = w = h = None
    if use_roi:
        roi = cv2.selectROI("Select ROI (press ENTER)", first, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        x, y, w, h = map(int, roi)
        if w == 0 or h == 0:
            x = y = w = h = None  # fall back to full frame

    ema_energy = None
    frame_idx = 0

    # Rewind to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if w is not None:
            roi_gray = gray_full[y:y+h, x:x+w]
            _, energy = apply_gabor_bank(roi_gray, kernels)
            # place into a zero map for visualization
            energy_full = np.zeros_like(gray_full, dtype=np.float32)
            energy_full[y:y+h, x:x+w] = energy
        else:
            _, energy_full = apply_gabor_bank(gray_full, kernels)

        # EMA smoothing for stability in video
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

    # Decide by file extension
    ext = os.path.splitext(input_path.lower())[1]
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        process_image(input_path, output_root, kernels)
    elif ext in [".mp4", ".avi", ".mov", ".mkv", ".m4v"]:
        base = os.path.splitext(output_root)[0]
        process_video(input_path, base, kernels, use_roi=use_roi_for_video)
    else:
        raise ValueError("Unsupported input type. Use an image (.jpg/.png/...) or video (.mp4/.avi/...).")


if __name__ == "__main__":
    main()
