import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, cast

import numpy as np
from numpy.typing import NDArray
import soundfile as sf
import librosa
from omegaconf import DictConfig
from scipy.signal import butter, filtfilt, lfilter
import pyloudnorm as pyln
import soxr

# ---------------- DSP ----------------


def apply_param_eq(y: NDArray[np.floating], sr: int, bands: List[Dict]) -> NDArray[np.floating]:
  """
  Boosts or cuts specific frequency bands
  """
  out = y
  for b in bands:
    t = b["type"].lower()
    if t == "peaking":
      B, A = biquad_peaking(b["f0"], b.get("q", 1.0), b["gain_db"], sr)
    elif t == "lowshelf":
      B, A = biquad_shelf(b["f0"], b["gain_db"], sr, high=False)
    elif t == "highshelf":
      B, A = biquad_shelf(b["f0"], b["gain_db"], sr, high=True)
    else:
      continue
    out = np.asarray(lfilter(B, A, out), dtype=np.float32)
  return out


@dataclass
class ExpanderCfg:
  threshold_db: float
  ratio: float
  attack_ms: float
  release_ms: float


def expander(y, sr, cfg: ExpanderCfg):
  """
  Background noise removal
  """
  eps = 1e-12
  level = 20*np.log10(np.maximum(np.abs(y), eps))
  under = cfg.threshold_db - level
  gain_db = np.where(under > 0, -under*(1 - 1/cfg.ratio), 0.0)
  atk = math.exp(-1.0/(cfg.attack_ms*0.001*sr))
  rel = math.exp(-1.0/(cfg.release_ms*0.001*sr))
  sm = np.zeros_like(gain_db)
  g = 0.0
  for i, gd in enumerate(gain_db):
    coeff = atk if gd < g else rel
    g = coeff*g + (1-coeff)*gd
    sm[i] = g
  return y * (10**(sm/20.0))


@dataclass
class CompressorCfg:
  threshold_db: float
  ratio: float
  attack_ms: float
  release_ms: float
  makeup_db: float


def compressor(y: np.ndarray, sr: int, cfg: CompressorCfg) -> np.ndarray:
  """
  Decreases loud/quiet contrast, which would make subtle swallow sounds more consistently loud
  """
  eps = 1e-12
  level = 20.0 * np.log10(np.maximum(np.abs(y), eps))
  over = level - cfg.threshold_db
  gain_red_db = np.where(over > 0.0, over - over / cfg.ratio, 0.0)
  atk = math.exp(-1.0 / (cfg.attack_ms * 0.001 * sr))
  rel = math.exp(-1.0 / (cfg.release_ms * 0.001 * sr))
  sm = np.zeros_like(gain_red_db)
  g = 0.0
  for i, gr in enumerate(gain_red_db):
    coeff = atk if gr > g else rel
    g = coeff * g + (1.0 - coeff) * gr
    sm[i] = g
  makeup = safe_db_to_lin(cfg.makeup_db)
  gain_lin = makeup * (10.0 ** (-sm / 20.0))
  return y * gain_lin


def spectral_gate(y: np.ndarray, sr: int, threshold_db: float = -40.0, freq_smoothing: float = 1.0) -> np.ndarray:
  """
  Supress unwanted spectral energy (the energy for a specific frequency), like breathy high frequencies or low rumble
  """
  n_fft = 1024
  hop = n_fft // 4
  win = librosa.filters.get_window("hann", n_fft, fftbins=True)
  S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=win)
  mag, phase = np.abs(S), np.angle(S)
  n_frames = max(1, int((0.5 * sr) / hop))
  noise_ref = np.median(mag[:, :n_frames], axis=1, keepdims=True)
  thr = noise_ref * safe_db_to_lin(threshold_db)
  if freq_smoothing > 1.0:
    k = int(freq_smoothing)
    thr = librosa.decompose.nn_filter(
        thr, aggregate=np.median, metric="cosine", width=k)
  mask = (mag > thr).astype(mag.dtype)
  soft = 0.5 + 0.5 * (mag - thr) / (np.maximum(mag, thr) + 1e-12)
  M = np.clip(soft, 0.0, 1.0) * mask
  S_hat = M * mag * np.exp(1j * phase)
  return librosa.istft(S_hat, hop_length=hop, window=win, length=len(y))


def apply_hpss(y: np.ndarray, mask_power: float, keep: str,
               percussive_gain_db: float, harmonic_gain_db: float,
               sr: int) -> np.ndarray:
  """
  Separates tonal and transient parts, swallows are often transient and percussive
  """
  S = librosa.stft(y)
  Hm, Pm = librosa.decompose.hpss(
      np.abs(S), mask=True, margin=(1.0, 1.0), power=mask_power)
  Y_h = librosa.istft(S * Hm)
  Y_p = librosa.istft(S * Pm)
  if keep == "percussive":
    y_out = Y_p * safe_db_to_lin(percussive_gain_db)
  elif keep == "harmonic":
    y_out = Y_h * safe_db_to_lin(harmonic_gain_db)
  else:
    y_out = Y_p * safe_db_to_lin(percussive_gain_db) + \
        Y_h * safe_db_to_lin(harmonic_gain_db)
  return librosa.util.fix_length(y_out, size=len(y))


def rms_envelope(y: np.ndarray, sr: int, win_ms: int, hop_ms: float) -> np.ndarray:
  """
  Frame-rate RMS (no Python loops).
  Swallows have recognizable start and end timing. However, issues could be that it differs a bit more based on the person
  """
  frame_length = max(1, int(round(sr * win_ms / 1000.0)))
  hop = max(1, int(round(sr * hop_ms / 1000.0)))
  rms = librosa.feature.rms(y=y, frame_length=frame_length,
                            hop_length=hop, center=True)[0]
  return rms.astype(np.float32)


def de_esser(y, sr, band_hz, threshold_db, ratio, attack_ms, release_ms):
  """
  Reduce speech-like transients
  """
  sibil = bandpass(y, sr, band_hz[0], band_hz[1])
  eps = 1e-12
  level = 20*np.log10(np.maximum(np.abs(sibil), eps))
  over = level - threshold_db
  red_db = np.where(over > 0, over - over/ratio, 0.0)
  atk = math.exp(-1.0/(attack_ms*0.001*sr))
  rel = math.exp(-1.0/(release_ms*0.001*sr))
  sm = np.zeros_like(red_db)
  g = 0.0
  for i, r in enumerate(red_db):
    coeff = atk if r > g else rel
    g = coeff*g + (1-coeff)*r
    sm[i] = g
  gain_lin = 10**(-sm/20.0)
  y_band = sibil * gain_lin
  y_resid = y - sibil
  return y_resid + y_band


# ---------------- I/O ----------------


def load_audio(path: str,
               expected_sr: Optional[int] = None,
               mono: bool = True) -> Tuple[np.ndarray, int]:
  try:
    y, sr = sf.read(path, always_2d=False)
  except RuntimeError:
    y, sr = librosa.load(path, sr=None, mono=False)
  y = pcm_to_float32(y)
  if mono and y.ndim == 2:
    y = y.mean(axis=1)
  if expected_sr is not None and sr != expected_sr:
    y = soxr.resample(y, sr, expected_sr)
    sr = expected_sr
  return np.ascontiguousarray(y, dtype=np.float32), int(sr)


def save_audio(path: str, y: np.ndarray, sr: int):
  sf.write(path, y, sr, subtype="PCM_16")


def pcm_to_float32(y: np.ndarray) -> np.ndarray:
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

# ---------------- DSP blocks ----------------


def safe_db_to_lin(db: float) -> float:
  return 10.0 ** (db / 20.0)


def _wn(x: float) -> float:
  return float(np.clip(x, 1e-6, 1 - 1e-6))


def dc_block(y: np.ndarray, alpha: float = 0.995) -> np.ndarray:
  """
  Removes the DC offset (a constant bias in the waveform). Without it, downstream RMS/envelope detectors can misinterpret the signal baseline. Useful because microphones and preamps often introduce a slight DC bias that masks subtle transient events like swallows.
  """
  x1 = 0.0
  y1 = 0.0
  out = np.empty_like(y)
  for i, x in enumerate(y):
    y0 = x - x1 + alpha * y1
    out[i] = y0
    x1, y1 = x, y0
  return out


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
  """
  Converts the signal to a target sampling rate. Ensures uniform input resolution for feature extraction and models
  """
  if sr_in == sr_out:
    return y
  return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq")


def butter_hp(y: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
  """
  Removes low-frequency rumble below a cutoff. Useful for eliminating mic handling noise, breathing artifacts, or low-frequency body movement that can obscure swallow transients
  """
  wn = _wn(cutoff / (0.5 * sr))
  b, a = cast(Tuple[np.ndarray, np.ndarray], butter(
      order, wn, btype="highpass", output="ba"))
  return filtfilt(b, a, y)


def butter_lp(y: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
  """
  Removes high-frequency content above a cutoff. Useful for discarding hiss
  """
  wn = _wn(cutoff / (0.5 * sr))
  b, a = cast(Tuple[np.ndarray, np.ndarray], butter(
      order, wn, btype="lowpass", output="ba"))
  return filtfilt(b, a, y)


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
  """
  Adjusts the whole waveform to a target perceptual loudness (LUFS)
  """
  meter = pyln.Meter(sr)
  loud = meter.integrated_loudness(y.astype(np.float64))
  # TODO: Look into this (Temp optimization)
  gain_db = np.clip(target_lufs - loud, -20.0, 20.0)  # cap boost/cut
  return y * (10.0 ** (gain_db / 20.0))


def bandpass(y, sr, f_lo, f_hi, order=4):
  lo = _wn(f_lo / (0.5 * sr))
  hi = _wn(f_hi / (0.5 * sr))
  if not (lo < hi):
    hi = _wn(min(0.999, lo + 1e-3))
  b, a = cast(Tuple[np.ndarray, np.ndarray], butter(
      order, [lo, hi], btype="band", output="ba"))
  return lfilter(b, a, y)


def biquad_peaking(f0, q, gain_db, sr):
  A = 10**(gain_db/40)
  w0 = 2*np.pi*f0/sr
  alpha = np.sin(w0)/(2*q)
  b0 = 1 + alpha*A
  b1 = -2*np.cos(w0)
  b2 = 1 - alpha*A
  a0 = 1 + alpha/A
  a1 = -2*np.cos(w0)
  a2 = 1 - alpha/A
  b = np.array([b0, b1, b2]) / a0
  a = np.array([1.0, a1/a0, a2/a0])
  return b, a


def biquad_shelf(f0, gain_db, sr, high=True, slope=1.0):
  A = 10**(gain_db/40)
  w0 = 2*np.pi*f0/sr
  alpha = np.sin(w0)/2 * np.sqrt((A + 1/A)*(1/slope - 1) + 2)
  cosw = np.cos(w0)
  if high:
    b0 = A*((A+1)+(A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1)+(A+1)*cosw)
    b2 = A*((A+1)+(A-1)*cosw - 2*np.sqrt(A)*alpha)
    a0 = (A+1)-(A-1)*cosw + 2*np.sqrt(A)*alpha
    a1 = 2*((A-1)-(A+1)*cosw)
    a2 = (A+1)-(A-1)*cosw - 2*np.sqrt(A)*alpha
  else:
    b0 = A*((A+1)-(A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = 2*A*((A-1)-(A+1)*cosw)
    b2 = A*((A+1)-(A-1)*cosw - 2*np.sqrt(A)*alpha)
    a0 = (A+1)+(A-1)*cosw + 2*np.sqrt(A)*alpha
    a1 = -2*((A-1)+(A+1)*cosw)
    a2 = (A+1)+(A-1)*cosw - 2*np.sqrt(A)*alpha
  b = np.array([b0, b1, b2]) / a0
  a = np.array([1.0, a1/a0, a2/a0])
  return b, a


def _sanitize_wave(y: np.ndarray, peak_clip: float = 1.0) -> np.ndarray:
  y = np.asarray(y, np.float64)  # keep headroom
  y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
  if y.size:
    p = float(np.max(np.abs(y)))
    if np.isfinite(p) and p > 0:
      y = y / max(1.0, p / peak_clip)  # prevent huge peaks
  y = np.clip(y, -peak_clip, peak_clip)
  return y.astype(np.float32, copy=False)


# ---------------- pipeline ----------------


def preprocess_waveform(y: np.ndarray, sr: int, cfg: DictConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
  if not getattr(cfg.preprocess, "enabled", True):
    return y.astype(np.float32), {}
  aux: Dict[str, np.ndarray] = {}
  for step in cfg.preprocess.pipeline:
    if step == "dc_block" and cfg.preprocess.get("dc_block", {}).get("enabled", False):
      y = dc_block(y, alpha=0.995)
    elif step == "resample" and sr != cfg.audio_io.model_sr:
      y = resample(y, sr_in=sr, sr_out=cfg.audio_io.model_sr)
      sr = cfg.audio_io.model_sr
    elif step == "highpass" and cfg.preprocess.highpass.enabled:
      y = butter_hp(y, sr, cutoff=cfg.preprocess.highpass.cutoff_hz,
                    order=cfg.preprocess.highpass.order)
    elif step == "lowpass" and cfg.preprocess.lowpass.enabled:
      y = butter_lp(y, sr, cutoff=cfg.preprocess.lowpass.cutoff_hz,
                    order=cfg.preprocess.lowpass.order)
    elif step == "loudness_norm" and cfg.preprocess.get("loudness_norm", {}).get("enabled", False):
      y = loudness_normalize(
          y, sr, target_lufs=cfg.preprocess.loudness_norm.target_lufs)
    elif step == "hpss" and cfg.preprocess.get("hpss", {}).get("enabled", False):
      y = apply_hpss(y, mask_power=float(cfg.preprocess.hpss.mask_power),
                     keep=str(cfg.preprocess.hpss.keep),
                     percussive_gain_db=float(
          cfg.preprocess.hpss.percussive_gain_db),
          harmonic_gain_db=float(cfg.preprocess.hpss.harmonic_gain_db), sr=sr)
    elif step == "spectral_gate" and cfg.preprocess.get("spectral_gate", {}).get("enabled", False):
      y = spectral_gate(y, sr, threshold_db=float(cfg.preprocess.spectral_gate.threshold_db),
                        freq_smoothing=float(cfg.preprocess.spectral_gate.freq_smoothing))
    elif step == "param_eq" and cfg.preprocess.get("param_eq", {}).get("enabled", False):
      y = apply_param_eq(y, sr, cfg.preprocess.param_eq.bands)
    elif step == "expander" and cfg.preprocess.get("expander", {}).get("enabled", False):
      ex = ExpanderCfg(float(cfg.preprocess.expander.threshold_db),
                       float(cfg.preprocess.expander.ratio),
                       float(cfg.preprocess.expander.attack_ms),
                       float(cfg.preprocess.expander.release_ms))
      y = expander(y, sr, ex)
    elif step == "compressor" and cfg.preprocess.get("compressor", {}).get("enabled", False):
      comp = CompressorCfg(float(cfg.preprocess.compressor.threshold_db),
                           float(cfg.preprocess.compressor.ratio),
                           float(cfg.preprocess.compressor.attack_ms),
                           float(cfg.preprocess.compressor.release_ms),
                           float(cfg.preprocess.compressor.makeup_db))
      y = compressor(y, sr, comp)
    elif step == "de_esser" and cfg.preprocess.get("de_esser", {}).get("enabled", False):
      y = de_esser(y, sr, list(cfg.preprocess.de_esser.band_hz),
                   float(cfg.preprocess.de_esser.threshold_db),
                   float(cfg.preprocess.de_esser.ratio),
                   float(cfg.preprocess.de_esser.attack_ms),
                   float(cfg.preprocess.de_esser.release_ms))
    elif step == "envelope_aux" and cfg.preprocess.get("envelope_aux", {}).get("enabled", True):
      env = rms_envelope(
          y,
          sr,
          win_ms=int(cfg.preprocess.envelope_aux.rms_win_ms),
          # <— use feature hop, not sample-by-sample
          hop_ms=float(cfg.features.hop_ms),
      )
      aux["envelope_rms"] = env
    elif step == "amplitude_guard":
      m = np.max(np.abs(y)) + 1e-12
      if m > 1.0:
        y = y / m
    y = _sanitize_wave(y)
  return y.astype(np.float32), aux
