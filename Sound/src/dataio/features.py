from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import librosa


@dataclass
class FeatCfg:
  type: str = "mel"
  sr: int = 16000
  n_fft: int = 512
  win_ms: float = 25.0
  hop_ms: float = 10.0
  n_mels: int = 128
  fmin: int = 20
  fmax: int = 8000
  log_eps: float = 1e-6
  deltas_enabled: bool = False
  deltas_order: int = 2
  aux_enabled: bool = True
  aux_flux: bool = True
  aux_centroid: bool = True
  aux_zcr: bool = True
  aux_perc_mask_mean: bool = True
  aux_env_rms: bool = True  # will use aux["envelope_rms"] if provided
  standardize: bool = True
  standardize_per_recording: bool = True
  calibration_secs: float = 10.0


@dataclass
class WindowCfg:
  chunk_sec: float = 2.0
  overlap: float = 0.5  # 50%
  pad_mode: str = "reflect"  # for waveform padding before STFT if needed


def _frame_params(cfg: FeatCfg) -> Tuple[int, int]:
  win = int(round(cfg.sr * cfg.win_ms / 1000.0))
  hop = int(round(cfg.sr * cfg.hop_ms / 1000.0))
  return win, hop


def _stft_mag(y: np.ndarray, cfg: FeatCfg) -> Tuple[np.ndarray, np.ndarray, int]:
  win, hop = _frame_params(cfg)
  S = librosa.stft(y, n_fft=cfg.n_fft, win_length=win,
                   hop_length=hop, window="hann", center=True)
  mag = np.abs(S).astype(np.float32)
  phase = np.angle(S).astype(np.float32)
  return mag, phase, hop


def _mel_log(mag: np.ndarray, cfg: FeatCfg) -> np.ndarray:
  mel_fb = librosa.filters.mel(
      sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
  mel = mel_fb @ (mag ** 2)  # power mel
  mel = np.log(mel + cfg.log_eps).astype(np.float32)
  return mel  # (n_mels, T)


def _delta_stack(M: np.ndarray, order: int) -> np.ndarray:
  chans = [M]
  d1 = librosa.feature.delta(M, order=1)
  chans.append(d1.astype(np.float32))
  if order >= 2:
    d2 = librosa.feature.delta(M, order=2)
    chans.append(d2.astype(np.float32))
  return np.stack(chans, axis=0)  # (C, F, T)


def _spectral_flux(mag: np.ndarray) -> np.ndarray:
  # Half-wave rectified frame-to-frame diff of normalized spectrum
  eps = 1e-12
  mag = mag / (np.sum(mag, axis=0, keepdims=True) + eps)
  diff = np.diff(mag, axis=1, prepend=mag[:, :1])
  flux = np.maximum(diff, 0.0).sum(axis=0)
  return flux.astype(np.float32)  # (T,)


def _spectral_centroid(mag: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
  freqs = np.linspace(0, sr/2, mag.shape[0], dtype=np.float32)
  num = (freqs[:, None] * mag).sum(axis=0)
  den = np.maximum(mag.sum(axis=0), 1e-12)
  return (num / den).astype(np.float32)  # (T,)


def _zcr(y: np.ndarray, cfg: FeatCfg, T: int) -> np.ndarray:
  win, hop = _frame_params(cfg)
  z = librosa.feature.zero_crossing_rate(
      y, frame_length=win, hop_length=hop, center=True)
  z = z.reshape(-1)
  if z.shape[0] != T:
    z = librosa.util.fix_length(z, size=T)
  return z.astype(np.float32)


def _perc_mask_mean(mag: np.ndarray) -> np.ndarray:
  # Use HPSS masks to get percussive mask, then mean over freq per frame
  H, P = librosa.decompose.hpss(mag, mask=True, margin=(1.0, 1.0), power=2.0)
  pm = np.clip(P, 0.0, 1.0)
  return pm.mean(axis=0).astype(np.float32)  # (T,)


def _align_env_to_frames(env: np.ndarray, hop: int, T: int) -> np.ndarray:
  centers = np.arange(T) * hop
  idx = np.clip(centers, 0, len(env) - 1).astype(int)
  return _safe_vec(env[idx])


def _standardize(M: np.ndarray, per_recording: bool, calibration_secs: float, cfg: FeatCfg) -> np.ndarray:
  if not per_recording:
    mu = np.median(M, axis=1, keepdims=True)
    sig = np.median(np.abs(M - mu), axis=1, keepdims=True) + 1e-6
    return (M - mu) / sig
  # use first N seconds as calibration
  _, hop = _frame_params(cfg)
  T_cal = int(round(calibration_secs * cfg.sr / hop))
  cal = M[:, :max(1, min(T_cal, M.shape[1]))]
  mu = np.median(cal, axis=1, keepdims=True)
  sig = np.median(np.abs(cal - mu), axis=1, keepdims=True) + 1e-6
  return (M - mu) / sig


def _safe_vec(v: np.ndarray) -> np.ndarray:
  v = np.asarray(v, dtype=np.float32)
  v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
  return np.clip(v, -1e6, 1e6)


def _fit_to_T(v: np.ndarray, T: int) -> np.ndarray:
  v = _safe_vec(v)
  if v.shape[0] != T:
    v = librosa.util.fix_length(v, size=T)
  return v


def extract_features(
    y: np.ndarray,
    sr: int,
    cfg: FeatCfg,
    aux: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, int, int]:
  """
  Returns X of shape (C, F, T), where F depends on feature type
  (e.g. n_mels for mel). C=1(+deltas)+aux_channels.
  """
  assert sr == cfg.sr, f"Expected sr={cfg.sr}, got {sr}"
  mag, _, hop = _stft_mag(y, cfg)

  if cfg.type == "mel":
    feat = _mel_log(mag, cfg)
  elif cfg.type == "spec":
    feat = np.log(mag + cfg.log_eps).astype(np.float32)
  else:
    raise ValueError(f"Unsupported feature type: {cfg.type}")

  if cfg.standardize:
    feat = _standardize(feat, cfg.standardize_per_recording,
                        cfg.calibration_secs, cfg)

  if cfg.deltas_enabled:
    base = _delta_stack(feat, cfg.deltas_order)
  else:
    base = feat[None, ...]

  T = feat.shape[1]
  chans: List[np.ndarray] = [base]

  if cfg.aux_enabled:
    aux_list = []
    if cfg.aux_flux:
      aux_list.append(_spectral_flux(mag))
    if cfg.aux_centroid:
      aux_list.append(_spectral_centroid(mag, cfg.sr, cfg.n_fft))
    if cfg.aux_zcr:
      aux_list.append(_zcr(y, cfg, T))
    if cfg.aux_perc_mask_mean:
      aux_list.append(_perc_mask_mean(mag))
    if cfg.aux_env_rms and aux is not None and "envelope_rms" in aux:
      aux_list.append(_align_env_to_frames(aux["envelope_rms"], hop, T))

    if aux_list:
      aux_list = [_fit_to_T(v, T) for v in aux_list]
      A = np.stack(aux_list, axis=0)
      A_tiled = np.repeat(A[:, None, :], feat.shape[0], axis=1)
      chans.append(A_tiled.astype(np.float32))

  X = np.concatenate(chans, axis=0).astype(np.float32)
  return X, feat.shape[0], hop


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
