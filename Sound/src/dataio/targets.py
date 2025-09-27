from typing import List, Tuple, Optional, Dict
import numpy as np


def events_to_frame_targets(
    events: List[Dict],          # [{"onset": s, "offset": s}|{"onset": s}]
    T: int,                      # total frames
    sr: int,
    hop: int,
    weak_dur_s: float = 0.4,     # use if offset missing
    soft_dilate_frames: int = 0,  # e.g., 2 frames -> onset bands
    min_event_frames: int = 0,   # e.g., 12 for ~120 ms @10 ms hop
) -> np.ndarray:
  """Return (T,) binary/soft targets."""
  y = np.zeros(T, dtype=np.float32)
  for e in events:
    t0 = int(round(e["onset"] * sr / hop))
    t1 = int(round(((e.get("offset", e["onset"] + weak_dur_s)) * sr) / hop))
    t0 = max(0, min(T - 1, t0))
    t1 = max(t0 + 1, min(T, t1))
    y[t0:t1] = 1.0

  if min_event_frames > 1 and y.any():
    # enforce minimum duration by dilating positives
    from scipy.ndimage import binary_closing
    y = binary_closing(y > 0, structure=np.ones(
        min_event_frames, int)).astype(np.float32)

  if soft_dilate_frames > 0 and y.any():
    # soft edges around onsets/offsets (triangular)
    edges = np.where(np.diff(np.pad(y, (1, 1))) != 0)[0]
    w = soft_dilate_frames
    soft = np.zeros_like(y)
    for idx in edges:
      for k in range(1, w+1):
        i1 = idx-k
        i2 = idx+k-1
        if 0 <= i1 < len(y):
          soft[i1] = np.maximum(soft[i1], 1.0 - k/(w+1))
        if 0 <= i2 < len(y):
          soft[i2] = np.maximum(soft[i2], 1.0 - k/(w+1))
    y = np.maximum(y, soft).astype(np.float32)
  return y


def slice_targets_to_windows(
    y_frames: np.ndarray,             # (T,)
    idxs: List[Tuple[int, int]],       # [(t0,t1),...]
) -> np.ndarray:
  return np.stack([y_frames[t0:t1] for (t0, t1) in idxs], axis=0)


def counts_from_events(
    events: List[Dict],
    idxs: List[Tuple[int, int]],
    sr: int,
    hop: int,
) -> np.ndarray:
  """Count events by onset falling in the window."""
  onsets_frames = [int(round(e["onset"] * sr / hop)) for e in events]
  counts = []
  for t0, t1 in idxs:
    c = sum(t0 <= f < t1 for f in onsets_frames)
    counts.append(c)
  return np.asarray(counts, dtype=np.float32)
