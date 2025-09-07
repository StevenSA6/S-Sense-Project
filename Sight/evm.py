import cv2
import numpy as np
from scipy.signal import butter, filtfilt

# --- params ---
input_path  = "C:\\Workspace\\ProgProj\\Eul\\data\\20250907_080326.mp4"
output_path = "C:\\Workspace\\ProgProj\\Eul\\data\\20250907_080326_OUT.mp4"
alpha       = 99.0           # amplification factor
fl, fh      = 0.8, 2.0       # band (Hz): 0.8–2.0 ~ heart-rate
pyr_scale   = 0.5            # downsample factor for speed (0.5 = half-size)
levels      = 0              # set > 0 to use a simple Gaussian pyramid (0 means none)
# ---------------

