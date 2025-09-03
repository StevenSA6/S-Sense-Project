# augment.py
import os
import random
from typing import Optional
import numpy as np
import librosa
from scipy.signal import fftconvolve
from preprocess import compressor, CompressorCfg, expander, ExpanderCfg, apply_param_eq, biquad_shelf
from omegaconf import DictConfig
from scipy.signal import lfilter


def _rand(p: float) -> bool: return random.random() < p
def _db2lin(db: float) -> float: return 10**(db/20)


def _fixlen(y: np.ndarray, n: int) -> np.ndarray:
  return librosa.util.fix_length(y, size=n)


def add_noise_snr(y: np.ndarray, snr_db: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
  n = noise if noise is not None else np.random.randn(
      len(y)).astype(np.float32)
  n = _fixlen(n, len(y))
  ps = np.mean(y**2) + 1e-12
  pn = ps / (_db2lin(snr_db)**2)
  n = n / (np.sqrt(np.mean(n**2) + 1e-12)) * np.sqrt(pn)
  return (y + n).astype(np.float32)


def _pick_noise(noise_bank: Optional[str]) -> Optional[np.ndarray]:
  if not noise_bank or not os.path.isdir(noise_bank):
    return None
  cands = [f for f in os.listdir(
      noise_bank) if f.lower().endswith((".wav", ".flac", ".ogg"))]
  if not cands:
    return None
  fp = os.path.join(noise_bank, random.choice(cands))
  n, sr = librosa.load(fp, sr=None, mono=True)
  return n.astype(np.float32)


def apply_ir(y: np.ndarray, ir_dir: Optional[str], wet_db: float = -12.0) -> np.ndarray:
  if not ir_dir or not os.path.isdir(ir_dir):
    return y
  irs = [f for f in os.listdir(
      ir_dir) if f.lower().endswith((".wav", ".flac", ".ogg"))]
  if not irs:
    return y
  ir, sr = librosa.load(os.path.join(
      ir_dir, random.choice(irs)), sr=None, mono=True)
  ir = ir / (np.max(np.abs(ir))+1e-12)
  wet = _db2lin(wet_db)
  out = fftconvolve(y, ir, mode="full")[:len(y)]
  return (y + wet*out).astype(np.float32)


def eq_tilt(y: np.ndarray, sr: int, db_per_octave: float) -> np.ndarray:
  # implement as opposing low/high shelves around 1 kHz
  f0 = 1000.0
  gain = db_per_octave * 2.0  # coarse mapping
  # low shelf +gain, high shelf -gain or vice versa
  B1, A1 = biquad_shelf(f0, gain, sr, high=False)
  B2, A2 = biquad_shelf(f0, -gain, sr, high=True)
  tmp1 = lfilter(B1, A1, y)
  tmp2 = lfilter(B2, A2, tmp1)
  assert isinstance(
      tmp2, np.ndarray), f"tmp2 should be of type NDArray but got f{type(tmp2)} instead"
  return tmp2.astype(np.float32)


def hpss_mix(y: np.ndarray, percussive_weight: float) -> np.ndarray:
  S = librosa.stft(y)
  Hm, Pm = librosa.decompose.hpss(
      np.abs(S), mask=True, margin=(1.0, 1.0), power=2.0)
  Y_h = librosa.istft(S * Hm)
  Y_p = librosa.istft(S * Pm)
  w = float(np.clip(percussive_weight, 0.0, 1.0))
  out = (1-w)*y + w*Y_p
  return _fixlen(out.astype(np.float32), len(y))


def stretch(y: np.ndarray, rate: float) -> np.ndarray:
  return _fixlen(librosa.effects.time_stretch(y, rate=rate), len(y))


def pitch(y: np.ndarray, sr: int, steps: float) -> np.ndarray:
  return _fixlen(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps), len(y))


def dry_food_sim(y: np.ndarray, sr: int, atten_db: float, dur_scale: float) -> np.ndarray:
  y2 = y.copy()
  # locate a transient by spectral flux
  S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
  flux = np.maximum(np.diff(S, axis=1, prepend=S[:, :1]), 0).sum(axis=0)
  t = int(np.argmax(flux))
  hop = 128
  t0 = max(0, t - int(0.10*sr/hop))
  t1 = min(S.shape[1]-1, t0 + int(0.30*sr/hop))
  i0, i1 = t0*hop, min(len(y2), t1*hop)
  seg = y2[i0:i1]
  if len(seg) > int(0.05*sr):
    seg2 = librosa.effects.time_stretch(seg, rate=max(0.5, dur_scale))
    seg2 = _db2lin(atten_db) * _fixlen(seg2, len(seg))
    y2[i0:i1] = seg2
  else:
    y2 *= _db2lin(atten_db)
  return y2.astype(np.float32)


def specaugment(mel: np.ndarray, sr: int, hop: int,
                time_mask_ms: int, time_masks: int,
                freq_mask_bins: int, freq_masks: int) -> np.ndarray:
  M = mel.copy()
  F, T = M.shape
  # time masks
  tmax = int(round((time_mask_ms/1000.0)*sr/hop))
  for _ in range(time_masks):
    w = random.randint(1, max(1, tmax))
    t0 = random.randint(0, max(0, T-w))
    M[:, t0:t0+w] = M[:, t0:t0+w].min()  # or 0
  # freq masks
  for _ in range(freq_masks):
    w = random.randint(1, max(1, freq_mask_bins))
    f0 = random.randint(0, max(0, F-w))
    M[f0:f0+w, :] = M[f0:f0+w, :].min()
  return M


def augment_waveform(y: np.ndarray, sr: int, cfg: DictConfig) -> np.ndarray:
  if not cfg.augment.enabled:
    return y
  a = cfg.augment

  # additive noise
  if _rand(a.prob.add_noise):
    snr = random.uniform(a.add_noise.snr_db_min, a.add_noise.snr_db_max)
    n = _pick_noise(getattr(a.add_noise, "noise_bank", None))
    y = add_noise_snr(y, snr_db=snr, noise=n)

  # IR convolution
  if _rand(a.prob.room_ir):
    y = apply_ir(y, getattr(a.room_ir, "ir_dir", None),
                 wet_db=a.room_ir.wet_db)

  # dynamic range
  if _rand(a.prob.dynamic_range):
    mode = a.dynamic_range.mode
    if mode == "random":
      mode = random.choice(["compress", "expand"])
    if mode == "compress":
      comp = CompressorCfg(a.dynamic_range.threshold_db,
                           random.uniform(max(1.1, a.dynamic_range.ratio_min), max(
                               1.2, a.dynamic_range.ratio_max)),
                           5, 80, 0.0)
      y = compressor(y, sr, comp)
    else:
      exp = ExpanderCfg(a.dynamic_range.threshold_db,
                        random.uniform(
                            1.1, max(1.2, a.dynamic_range.ratio_max)),
                        5, 80)
      y = expander(y, sr, exp)

  # EQ tilt
  if _rand(a.prob.eq_tilt):
    tilt = random.uniform(a.eq_tilt.db_per_octave_min,
                          a.eq_tilt.db_per_octave_max)
    y = eq_tilt(y, sr, tilt)

  # HPSS mix
  if _rand(a.prob.hpss_mix):
    w = random.uniform(a.hpss_mix.percussive_weight_min,
                       a.hpss_mix.percussive_weight_max)
    y = hpss_mix(y, w)

  # time stretch
  if _rand(a.prob.time_stretch):
    rate = random.uniform(a.time_stretch.min, a.time_stretch.max)
    y = stretch(y, rate)

  # pitch shift
  if _rand(a.prob.pitch_shift):
    steps = random.uniform(a.pitch_shift.semitones_min,
                           a.pitch_shift.semitones_max)
    y = pitch(y, sr, steps)

  # dry-food sim
  if _rand(a.prob.dry_food_sim):
    att = random.uniform(a.dry_food_sim.attenuate_db_min,
                         a.dry_food_sim.attenuate_db_max)
    sc = random.uniform(a.dry_food_sim.duration_scale_min,
                        a.dry_food_sim.duration_scale_max)
    y = dry_food_sim(y, sr, att, sc)

  # amplitude guard
  peak = np.max(np.abs(y))+1e-12
  if peak > 1.0:
    y = y/peak
  return y.astype(np.float32)
