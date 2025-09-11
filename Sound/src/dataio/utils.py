from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import soxr
from pathlib import Path
from dataio.dataset import load_events_csv


def load_audio(path: str,
               expected_sr: Optional[int] = None,
               mono: bool = True) -> Tuple[np.ndarray, int]:
  try:
    y, sr = sf.read(path, always_2d=False)
  except RuntimeError:
    y, sr = librosa.load(path, sr=None, mono=False)
  y = _pcm_to_float32(y)
  if mono and y.ndim == 2:
    y = y.mean(axis=1)
  if expected_sr is not None and sr != expected_sr:
    y = soxr.resample(y, sr, expected_sr)
    sr = expected_sr
  return np.ascontiguousarray(y, dtype=np.float32), int(sr)


def save_audio(path: str, y: np.ndarray, sr: int):
  sf.write(path, y, sr, subtype="PCM_16")


def scan_audio_dir(audio_dir: str, timestamps_dir: str) -> list[dict]:
  exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
  audio_root = Path(audio_dir)
  items = []
  for p in sorted(audio_root.rglob("*")):
    if p.suffix.lower() not in exts:
      continue
    subj = p.parent.name  # parent folder = subject
    rel = p.relative_to(audio_root).with_suffix(".csv")
    csv_path = Path(timestamps_dir) / rel
    events = load_events_csv(csv_path) if csv_path.exists() else []
    items.append({"path": str(p), "subject": subj, "events": events})
  return items


@dataclass
class WindowCfg:
  chunk_sec: float = 2.0
  overlap: float = 0.5  # 50%
  pad_mode: str = "reflect"  # for waveform padding before STFT if needed


def window_into_chunks(
    X: np.ndarray,     # (C,F,T)
    sr: int,
    hop: int,
    wcfg: WindowCfg
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
  """
  Chunk along time dimension with 50% overlap.
  Returns windows W of shape (N, C, F, Tw) and [(t0,t1) frame indices].
  """
  C, F, T = X.shape
  frames_per_chunk = int(round(wcfg.chunk_sec * sr / hop))
  step = max(1, int(round(frames_per_chunk * (1.0 - wcfg.overlap))))
  if frames_per_chunk <= 0:
    raise ValueError("frames_per_chunk <= 0")

  # pad reflect to cover last partial window
  pad = (0, 0)
  need = (((T - frames_per_chunk) % step) != 0)
  if need:
    remainder = (T - frames_per_chunk) % step
    pad_frames = step - remainder
    X = np.pad(X, ((0, 0), (0, 0), (0, pad_frames)), mode="reflect")
    T = X.shape[2]

  windows = []
  idxs: List[Tuple[int, int]] = []
  for t0 in range(0, T - frames_per_chunk + 1, step):
    t1 = t0 + frames_per_chunk
    windows.append(X[:, :, t0:t1])
    idxs.append((t0, t1))
  W = np.stack(windows, axis=0) if windows else np.empty(
      (0, C, F, frames_per_chunk), dtype=X.dtype)
  return W.astype(np.float32), idxs


# ---------------- Helpers ----------------

def _pcm_to_float32(y: np.ndarray) -> np.ndarray:
  """
  ensures downstream code always gets np.float32 audio arrays scaled to [-1, 1].
  Makes all the loaded audio consistent - because multiple libraries are used
  """
  if np.issubdtype(y.dtype, np.integer):
    if y.dtype == np.uint8:
      return ((y.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    return (y.astype(np.float32) / float(np.iinfo(y.dtype).max)).clip(-1.0, 1.0)
  if np.issubdtype(y.dtype, np.floating):
    y32 = y.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y32))) if y32.size else 0.0
    return y32 if peak == 0.0 or peak <= 1.0 else (y32 / peak)
  raise TypeError(f"Unsupported dtype: {y.dtype!r}")
