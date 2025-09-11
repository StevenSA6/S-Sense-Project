from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import soxr


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
