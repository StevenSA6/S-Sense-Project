from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import librosa
from scipy.signal import find_peaks
from omegaconf import DictConfig

from dataio.preprocess import load_audio, preprocess_waveform


def _frame_params(sr: int, win_ms: float, hop_ms: float) -> Tuple[int, int]:
  win = int(round(sr * win_ms / 1000.0))
  hop = int(round(sr * hop_ms / 1000.0))
  return win, hop


def _stft_mag(y: np.ndarray, sr: int, n_fft: int, win_ms: float, hop_ms: float) -> Tuple[np.ndarray, int]:
  win, hop = _frame_params(sr, win_ms, hop_ms)
  S = librosa.stft(y, n_fft=n_fft, win_length=win, hop_length=hop,
                   window="hann", center=True)
  mag = np.abs(S).astype(np.float32)
  return mag, hop


def _spectral_flux(mag: np.ndarray) -> np.ndarray:
  mag = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-12)
  diff = np.diff(mag, axis=1, prepend=mag[:, :1])
  flux = np.maximum(diff, 0.0).sum(axis=0)
  return flux.astype(np.float32)


def _percussive_energy(mag: np.ndarray) -> np.ndarray:
  _, P = librosa.decompose.hpss(mag, margin=(1.0, 1.0))
  return P.sum(axis=0).astype(np.float32)


def _highband_energy(mag: np.ndarray, sr: int) -> np.ndarray:
  F = mag.shape[0]
  cutoff = sr / 4.0
  bin_cut = int(round(cutoff / (sr / 2.0) * F))
  hb = mag[bin_cut:, :].sum(axis=0)
  return hb.astype(np.float32)


def _crest_factor(y: np.ndarray, win: int, hop: int, T: int) -> np.ndarray:
  frames = librosa.util.frame(y, frame_length=win, hop_length=hop)
  peak = np.max(np.abs(frames), axis=0)
  rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-9)
  cf = peak / (rms + 1e-9)
  if cf.shape[0] != T:
    cf = librosa.util.fix_length(cf, size=T)
  return cf.astype(np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
  m = np.median(v)
  s = np.median(np.abs(v - m)) + 1e-6
  return (v - m) / s


def detect_path(path: str, cfg: DictConfig) -> Dict:
  y, sr_raw = load_audio(path, expected_sr=None, mono=True)
  y_p, _ = preprocess_waveform(y, sr_raw, cfg)
  sr = cfg.audio_io.model_sr
  win_ms = cfg.features.win_ms
  hop_ms = cfg.features.hop_ms
  n_fft = getattr(cfg.features, "n_fft", 512)
  mag, hop = _stft_mag(y_p, sr, n_fft, win_ms, hop_ms)
  feats: List[np.ndarray] = []
  bcfg = cfg.baseline.features
  if getattr(bcfg, "spectral_flux", False):
    feats.append(_normalize(_spectral_flux(mag)))
  if getattr(bcfg, "percussive_energy", False):
    feats.append(_normalize(_percussive_energy(mag)))
  if getattr(bcfg, "highband_energy", False):
    feats.append(_normalize(_highband_energy(mag, sr)))
  if getattr(bcfg, "crest_factor", False):
    win, _ = _frame_params(sr, win_ms, hop_ms)
    feats.append(_normalize(_crest_factor(y_p, win, hop, mag.shape[1])))
  if not feats:
    feats.append(_normalize(mag.sum(axis=0)))
  env = np.mean(np.stack(feats, axis=0), axis=0)
  dist = int(round(cfg.baseline.detector.min_gap_sec * sr / hop))
  peaks, _ = find_peaks(env, distance=max(1, dist),
                        prominence=cfg.baseline.detector.peak_prominence)
  onsets = peaks * hop / sr
  segs = [(int(p), int(p + 1)) for p in peaks]
  return {
      "posteriors": env,
      "segments_frames": segs,
      "onsets_s": onsets.tolist(),
      "offsets_s": onsets.tolist(),
      "count": len(onsets),
      "hop": hop,
      "sr": sr,
  }
