# preprocess.py
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, cast

import numpy as np
from numpy.typing import NDArray
import soundfile as sf
import librosa
from omegaconf import DictConfig, OmegaConf
from scipy.signal import butter, filtfilt, lfilter
import pyloudnorm as pyln
import soundfile as sf
import soxr


def main():
  p = argparse.ArgumentParser(description="Preprocess audio per config")
  p.add_argument("--config", type=str, required=True,
                 help="configs/config.yaml")
  p.add_argument("--in", dest="inp", type=str,
                 required=True, help="file or directory")
  p.add_argument("--out", dest="out", type=str, required=True,
                 help="output file or directory")
  p.add_argument("--save-aux", action="store_true",
                 help="save envelope as .npy alongside audio")
  args = p.parse_args()

  cfg = OmegaConf.load(args.config)
  assert isinstance(cfg, DictConfig), f"Expected DictConfig, got {type(cfg)}"

  in_path = args.inp
  out_path = args.out

  def process_one(ip: str, op: str):
    y, sr = load_audio(ip)
    y_p, aux = preprocess_waveform(y, sr, cfg)
    # ensure output sr equals model_sr
    save_audio(op, y_p, sr=cfg.audio_io.model_sr)
    if args.save_aux and "envelope_rms" in aux:
      np.save(os.path.splitext(op)[0] + "_env.npy", aux["envelope_rms"])

  if os.path.isdir(in_path):
    os.makedirs(out_path, exist_ok=True)
    for fn in os.listdir(in_path):
      if not fn.lower().endswith((".wav", ".flac", ".ogg")):
        continue
      ip = os.path.join(in_path, fn)
      op = os.path.join(out_path, os.path.splitext(fn)[0] + "_prep.wav")
      process_one(ip, op)
  else:
    # single file
    if os.path.isdir(out_path):
      base = os.path.splitext(os.path.basename(in_path))[0] + "_prep.wav"
      op = os.path.join(out_path, base)
    else:
      op = out_path
    os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
    process_one(in_path, op)


@dataclass
class ExpanderCfg:
  threshold_db: float
  ratio: float
  attack_ms: float
  release_ms: float


def expander(y, sr, cfg: ExpanderCfg):
  eps = 1e-12
  level = 20*np.log10(np.maximum(np.abs(y), eps))
  under = cfg.threshold_db - level
  # gain increase when below threshold, scaled by ratio
  gain_db = np.where(under > 0, -under*(1 - 1/cfg.ratio), 0.0)
  atk = math.exp(-1.0/(cfg.attack_ms*0.001*sr))
  rel = math.exp(-1.0/(cfg.release_ms*0.001*sr))
  sm = np.zeros_like(gain_db)
  g = 0.0
  for i, gd in enumerate(gain_db):
    coeff = atk if gd < g else rel
    g = coeff*g + (1-coeff)*gd
    sm[i] = g
  gain_lin = 10**(sm/20.0)
  return y * gain_lin


def bandpass(y, sr, f_lo, f_hi, order=4):
  lo = _wn(f_lo / (0.5 * sr))
  hi = _wn(f_hi / (0.5 * sr))
  if not (lo < hi):
    # ensure strict ordering if config is bad
    hi = _wn(min(0.999, lo + 1e-3))
  res = butter(order, [lo, hi], btype="band", output="ba")
  b, a = cast(Tuple[np.ndarray, np.ndarray], res)
  return lfilter(b, a, y)


def de_esser(y, sr, band_hz, threshold_db, ratio, attack_ms, release_ms):
  sibil = bandpass(y, sr, band_hz[0], band_hz[1])
  # sidechain level
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
  # apply only to the band via simple split-sum
  y_band = sibil * gain_lin
  y_resid = y - sibil
  return y_resid + y_band


def to_mono(y: np.ndarray) -> np.ndarray:
  if y.ndim == 1:
    return y
  return np.mean(y, axis=0)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
  if sr_in == sr_out:
    return y
  # Prefer soxr if available through librosa
  return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq")


def safe_db_to_lin(db: float) -> float:
  return 10.0 ** (db / 20.0)


def dc_block(y: np.ndarray, alpha: float = 0.995) -> np.ndarray:
  """Simple DC blocking filter."""
  x1 = 0.0
  y1 = 0.0
  out = np.empty_like(y)
  for i, x in enumerate(y):
    y0 = x - x1 + alpha * y1
    out[i] = y0
    x1, y1 = x, y0
  return out


def _wn(x: float) -> float:
  # clamp to (0,1) open interval
  return float(np.clip(x, 1e-6, 1 - 1e-6))


def butter_hp(y: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
  wn = _wn(cutoff / (0.5 * sr))
  res = butter(order, wn, btype="highpass", output="ba")
  b, a = cast(Tuple[np.ndarray, np.ndarray], res)
  return filtfilt(b, a, y)


def butter_lp(y: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
  wn = _wn(cutoff / (0.5 * sr))
  res = butter(order, wn, btype="lowpass", output="ba")
  b, a = cast(Tuple[np.ndarray, np.ndarray], res)
  return filtfilt(b, a, y)


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
  meter = pyln.Meter(sr)  # EBU R128
  loud = meter.integrated_loudness(y.astype(np.float64))
  gain_db = target_lufs - loud
  return y * safe_db_to_lin(gain_db)


def apply_hpss(y: np.ndarray, mask_power: float, keep: str,
               percussive_gain_db: float, harmonic_gain_db: float,
               sr: int) -> np.ndarray:
  # librosa HPSS works on STFT magnitude; use default kernel
  H, P = librosa.decompose.hpss(librosa.stft(
      y), mask=True, margin=(1.0, 1.0), power=mask_power)
  Sh, Sp = H, P  # masks
  S = librosa.stft(y)
  Y_h = librosa.istft(S * Sh)
  Y_p = librosa.istft(S * Sp)
  if keep == "percussive":
    y_out = Y_p * safe_db_to_lin(percussive_gain_db)
  elif keep == "harmonic":
    y_out = Y_h * safe_db_to_lin(harmonic_gain_db)
  else:  # both
    y_out = Y_p * safe_db_to_lin(percussive_gain_db) + \
        Y_h * safe_db_to_lin(harmonic_gain_db)
  return librosa.util.fix_length(y_out, size=len(y))


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


def apply_param_eq(
    y: NDArray[np.floating],
    sr: int,
    bands: List[Dict]
) -> NDArray[np.floating]:
  """
  Apply a stack of parametric EQ filters to the waveform.
  Each band is a dict with keys:
    - type: 'peaking' | 'lowshelf' | 'highshelf'
    - f0: center frequency
    - q: quality factor (for peaking)
    - gain_db: gain in dB
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


def spectral_gate(y: np.ndarray, sr: int,
                  threshold_db: float = -40.0,
                  freq_smoothing: float = 1.0) -> np.ndarray:
  """
  Simple noise gate in STFT domain.
  Estimate noise floor from first 0.5 s. Attenuate bins below threshold.
  """
  n_fft = 1024
  hop = n_fft // 4
  win = librosa.filters.get_window("hann", n_fft, fftbins=True)
  S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=win)
  mag, phase = np.abs(S), np.angle(S)

  # noise profile from first half-second
  n_frames = max(1, int((0.5 * sr) / hop))
  noise_ref = np.median(mag[:, :n_frames], axis=1, keepdims=True)  # (F,1)

  # threshold per-bin
  thr = noise_ref * safe_db_to_lin(threshold_db)
  if freq_smoothing > 1.0:
    k = int(freq_smoothing)
    thr = librosa.decompose.nn_filter(
        thr, aggregate=np.median, metric="cosine", width=k)

  mask = (mag > thr).astype(mag.dtype)
  # Soft mask
  soft = 0.5 + 0.5 * (mag - thr) / (np.maximum(mag, thr) + 1e-12)
  M = np.clip(soft, 0.0, 1.0) * mask
  S_hat = M * mag * np.exp(1j * phase)
  y_hat = librosa.istft(S_hat, hop_length=hop, window=win, length=len(y))
  return y_hat


@dataclass
class CompressorCfg:
  threshold_db: float
  ratio: float
  attack_ms: float
  release_ms: float
  makeup_db: float


def compressor(y: np.ndarray, sr: int, cfg: CompressorCfg) -> np.ndarray:
  """
  Feed-forward compressor with level detector in dBFS and smoothing.
  """
  eps = 1e-12
  # Level in dBFS using absolute value
  level = 20.0 * np.log10(np.maximum(np.abs(y), eps))
  # Gain computer
  over = level - cfg.threshold_db
  gain_red_db = np.where(over > 0.0, over - over / cfg.ratio, 0.0)
  # Smooth gain reduction with attack/release
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


def rms_envelope(y: np.ndarray, sr: int, win_ms: int, release_ms: int) -> np.ndarray:
  win = max(1, int(sr * win_ms / 1000))
  # simple moving RMS
  pad = win // 2
  y2 = np.pad(y**2, (pad, pad), mode="reflect")
  kernel = np.ones(win) / win
  power = np.convolve(y2, kernel, mode="valid")
  rms = np.sqrt(np.maximum(power, 1e-12))
  # apply release smoothing
  rel = math.exp(-1.0 / (release_ms * 0.001 * sr))
  out = np.zeros_like(rms)
  v = 0.0
  for i, r in enumerate(rms):
    v = max(r, rel * v + (1 - rel) * r)
    out[i] = v
  return out


def preprocess_waveform(
    y: np.ndarray,
    sr: int,
    cfg: DictConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
  aux: Dict[str, np.ndarray] = {}

  # iterate over pipeline steps
  for step in cfg.preprocess.pipeline:
    if step == "dc_block" and cfg.preprocess.get("dc_block", {}).get("enabled", False):
      y = dc_block(y, alpha=0.995)

    elif step == "resample" and sr != cfg.audio_io.model_sr:
      y = resample(y, sr_in=sr, sr_out=cfg.audio_io.model_sr)
      sr = cfg.audio_io.model_sr

    elif step == "highpass" and cfg.preprocess.highpass.enabled:
      y = butter_hp(y, sr,
                    cutoff=cfg.preprocess.highpass.cutoff_hz,
                    order=cfg.preprocess.highpass.order)

    elif step == "lowpass" and cfg.preprocess.lowpass.enabled:
      y = butter_lp(y, sr,
                    cutoff=cfg.preprocess.lowpass.cutoff_hz,
                    order=cfg.preprocess.lowpass.order)

    elif step == "loudness_norm" and cfg.preprocess.get("loudness_norm", {}).get("enabled", False):
      y = loudness_normalize(
          y, sr, target_lufs=cfg.preprocess.loudness_norm.target_lufs)

    elif step == "hpss" and cfg.preprocess.get("hpss", {}).get("enabled", False):
      y = apply_hpss(
          y,
          mask_power=float(cfg.preprocess.hpss.mask_power),
          keep=str(cfg.preprocess.hpss.keep),
          percussive_gain_db=float(cfg.preprocess.hpss.percussive_gain_db),
          harmonic_gain_db=float(cfg.preprocess.hpss.harmonic_gain_db),
          sr=sr,
      )

    elif step == "spectral_gate" and cfg.preprocess.get("spectral_gate", {}).get("enabled", False):
      y = spectral_gate(
          y, sr,
          threshold_db=float(cfg.preprocess.spectral_gate.threshold_db),
          freq_smoothing=float(cfg.preprocess.spectral_gate.freq_smoothing),
      )

    elif step == "param_eq" and cfg.preprocess.get("param_eq", {}).get("enabled", False):
      y = apply_param_eq(y, sr, cfg.preprocess.param_eq.bands)

    elif step == "expander" and cfg.preprocess.get("expander", {}).get("enabled", False):
      ex = ExpanderCfg(
          threshold_db=float(cfg.preprocess.expander.threshold_db),
          ratio=float(cfg.preprocess.expander.ratio),
          attack_ms=float(cfg.preprocess.expander.attack_ms),
          release_ms=float(cfg.preprocess.expander.release_ms),
      )
      y = expander(y, sr, ex)

    elif step == "compressor" and cfg.preprocess.get("compressor", {}).get("enabled", False):
      comp = CompressorCfg(
          threshold_db=float(cfg.preprocess.compressor.threshold_db),
          ratio=float(cfg.preprocess.compressor.ratio),
          attack_ms=float(cfg.preprocess.compressor.attack_ms),
          release_ms=float(cfg.preprocess.compressor.release_ms),
          makeup_db=float(cfg.preprocess.compressor.makeup_db),
      )
      y = compressor(y, sr, comp)

    elif step == "de_esser" and cfg.preprocess.get("de_esser", {}).get("enabled", False):
      y = de_esser(
          y, sr,
          band_hz=list(cfg.preprocess.de_esser.band_hz),
          threshold_db=float(cfg.preprocess.de_esser.threshold_db),
          ratio=float(cfg.preprocess.de_esser.ratio),
          attack_ms=float(cfg.preprocess.de_esser.attack_ms),
          release_ms=float(cfg.preprocess.de_esser.release_ms),
      )

    elif step == "envelope_aux" and cfg.preprocess.get("envelope_aux", {}).get("enabled", True):
      env = rms_envelope(
          y,
          sr,
          win_ms=int(cfg.preprocess.envelope_aux.rms_win_ms),
          release_ms=int(cfg.preprocess.envelope_aux.release_ms),
      )
      aux["envelope_rms"] = env

    elif step == "amplitude_guard":
      max_abs = np.max(np.abs(y)) + 1e-12
      if max_abs > 1.0:
        y = y / max_abs

  return y.astype(np.float32), aux


def pcm_to_float32(y: np.ndarray) -> np.ndarray:
  if np.issubdtype(y.dtype, np.integer):
    if y.dtype == np.uint8:
      return ((y.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    return (y.astype(np.float32) / float(np.iinfo(y.dtype).max)).clip(-1.0, 1.0)
  if np.issubdtype(y.dtype, np.floating):
    y32 = y.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y32))) if y32.size else 0.0
    return y32 if peak == 0.0 or peak <= 1.0 else (y32 / peak)
  raise TypeError(f"Unsupported dtype: {y.dtype!r}")


def load_audio(path: str,
               expected_sr: Optional[int] = None,
               mono: bool = True) -> Tuple[np.ndarray, int]:
  try:
    y, sr = sf.read(path, always_2d=False)
  except RuntimeError:
    # fallback for MP3/other formats
    y, sr = librosa.load(path, sr=None, mono=False)
  y = pcm_to_float32(y)

  assert isinstance(sr, int), f"Expected type int but got type: {type(sr)}"
  if mono and y.ndim == 2:
    y = y.mean(axis=1)

  if expected_sr is not None and sr != expected_sr:
    y = soxr.resample(y, sr, expected_sr)
    sr = expected_sr

  return np.ascontiguousarray(y, dtype=np.float32), sr


def save_audio(path: str, y: np.ndarray, sr: int):
  sf.write(path, y, sr, subtype="PCM_16")


if __name__ == "__main__":
  main()
