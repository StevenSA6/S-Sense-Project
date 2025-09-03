# dataset.py
from __future__ import annotations
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import librosa
import soundfile as sf

import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig
# Your existing modules
from preprocess import load_audio, preprocess_waveform
from features import FeatCfg, WindowCfg, extract_features, window_into_chunks
from augment import augment_waveform, specaugment
from targets import events_to_frame_targets, slice_targets_to_windows, counts_from_events


# ---------- manifest utilities ----------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
  """Each line: {"path": "...", "subject": "S01", "events":[{"onset":1.23,"offset":1.55}, ...]}"""
  out = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      out.append(json.loads(line))
  return out


# ---------- dataset ----------

@dataclass
class DatasetConfig:
  cfg: DictConfig
  train: bool


class SwallowWindowDataset(Dataset):
  """
  Returns per-window samples ready for CRNN:
    x : (C, F, Tw) float32
    y : (Tw,) float32 frame targets
    count : () float32 window count target (onset-in-window)
  Uses on-the-fly waveform augmentations only when train=True.
  """

  def __init__(self, items: List[Dict[str, Any]], ds_cfg: DatasetConfig):
    self.items = items
    self.cfg = ds_cfg.cfg
    self.train = ds_cfg.train

    # feature cfg mapping from Hydra config
    fcfg = self.cfg.features
    self.feat_cfg = FeatCfg(
        sr=self.cfg.audio_io.model_sr,
        n_fft=getattr(fcfg, "n_fft", 512),
        win_ms=fcfg.win_ms,
        hop_ms=fcfg.hop_ms,
        n_mels=fcfg.n_mels,
        fmin=fcfg.fmin,
        fmax=fcfg.fmax,
        log_eps=getattr(fcfg, "log_eps", 1e-6),
        deltas_enabled=fcfg.deltas.enabled,
        deltas_order=fcfg.deltas.order,
        aux_enabled=fcfg.aux_channels.enabled,
        aux_flux=fcfg.aux_channels.spectral_flux,
        aux_centroid=fcfg.aux_channels.centroid,
        aux_zcr=fcfg.aux_channels.zcr,
        aux_perc_mask_mean=fcfg.aux_channels.percussive_mask_mean,
        aux_env_rms=fcfg.aux_channels.envelope_rms,
        standardize=fcfg.standardize.enabled,
        standardize_per_recording=fcfg.standardize.per_recording,
        calibration_secs=fcfg.standardize.calibration_secs,
    )
    self.win_cfg = WindowCfg(
        chunk_sec=self.cfg.windowing.chunk_sec,
        overlap=self.cfg.windowing.overlap,
        pad_mode=self.cfg.windowing.pad_mode,
    )

    # Build an index of (file_idx, lazy_window) without precomputing audio.
    # We estimate frames-per-chunk to know how many windows each file will yield after feature extraction.
    win, hop = _frame_params_ms(
        self.feat_cfg.sr, self.feat_cfg.win_ms, self.feat_cfg.hop_ms)
    self.frames_per_chunk = int(
        round(self.win_cfg.chunk_sec * self.feat_cfg.sr / hop))
    self.hop = hop  # save for target building
    # (file_idx, window_ordinal) — ordinal resolved on the fly
    self.index: List[Tuple[int, int]] = []

    for i, it in enumerate(self.items):
      n_frames = _estimate_frames(
          it["path"], target_sr=self.feat_cfg.sr, win=win, hop=hop)
      step = max(
          1, int(round(self.frames_per_chunk * (1.0 - self.win_cfg.overlap))))
      if n_frames < self.frames_per_chunk:
        n_windows = 1
      else:
        n_windows = 1 + math.floor((n_frames - self.frames_per_chunk) / step)
      for w in range(n_windows):
        self.index.append((i, w))

  def __len__(self) -> int:
    return len(self.index)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    file_idx, w_ord = self.index[idx]
    item = self.items[file_idx]

    # Load and DSP preprocess to model_sr
    y, sr_raw = load_audio(item["path"], expected_sr=None, mono=True)
    # respects pipeline + resample
    y_p, aux = preprocess_waveform(y, sr_raw, self.cfg)

    # On-the-fly waveform augmentation (train only)
    if self.train and self.cfg.augment.enabled:
      y_p = augment_waveform(y_p, self.cfg.audio_io.model_sr, self.cfg)

    # Features for full clip (T frames)
    X, F, hop = extract_features(
        y_p, self.cfg.audio_io.model_sr, self.feat_cfg, aux=aux)

    # Optional SpecAugment on base mel channel (train only)
    if self.train and self.cfg.augment.enabled and random.random() < self.cfg.augment.prob.specaugment:
      X[0] = specaugment(
          X[0], self.cfg.audio_io.model_sr, hop,
          self.cfg.augment.specaugment.time_mask_ms,
          self.cfg.augment.specaugment.time_masks,
          self.cfg.augment.specaugment.freq_mask_bins,
          self.cfg.augment.specaugment.freq_masks,
      )

    # Window the full sequence, then pick the requested window ordinal
    W, idxs = window_into_chunks(
        X, self.cfg.audio_io.model_sr, hop, self.win_cfg)
    if not idxs:  # degenerate case: force one window
      W = X[:, :, :self.frames_per_chunk][None, ...]
      idxs = [(0, min(self.frames_per_chunk, X.shape[-1]))]
    w_ord = min(w_ord, len(idxs)-1)
    x = W[w_ord]  # (C,F,Tw)
    t0, t1 = idxs[w_ord]

    # Targets
    # list of {"onset": s, "offset": s} or {"onset": s}
    events = item.get("events", [])
    weak_dur = float(self.cfg.labels.get("weak_dur_sec", 0.4)
                     )  # fallback if not in config
    soft_band = int(self.cfg.labels.get("positive_dilation_frames", 0))
    min_frames = int(round(self.cfg.labels.min_event_sec *
                     self.cfg.audio_io.model_sr / hop)) if "min_event_sec" in self.cfg.labels else 0

    y_frames_full = events_to_frame_targets(
        events=events,
        T=X.shape[-1],
        sr=self.cfg.audio_io.model_sr,
        hop=hop,
        weak_dur_s=weak_dur,
        soft_dilate_frames=soft_band,
        min_event_frames=min_frames,
    )
    y_win = y_frames_full[t0:t1]

    count_win = counts_from_events(
        events, [(t0, t1)], sr=self.cfg.audio_io.model_sr, hop=hop)[0]

    return {
        "x": torch.from_numpy(x),                    # (C,F,Tw)
        "y": torch.from_numpy(y_win).float(),        # (Tw,)
        "count": torch.tensor(count_win, dtype=torch.float32),  # ()
    }


def _frame_params_ms(sr: int, win_ms: float, hop_ms: float) -> Tuple[int, int]:
  win = int(round(sr * win_ms / 1000.0))
  hop = int(round(sr * hop_ms / 1000.0))
  return win, hop


def _estimate_frames(path: str, target_sr: int, win: int, hop: int) -> int:
  """Cheap duration probe to estimate STFT frame count."""
  try:
    info = sf.info(path)
    n_samples = info.frames
    sr_in = info.samplerate
  except Exception:
    # fallback: load header via librosa
    y, sr_in = librosa.load(path, sr=None, mono=True, duration=1.0)
    # skip precise load; we can do an OS stat as last resort
    n_samples = int(librosa.get_duration(filename=path) * sr_in)
  # estimate post-resample samples
  n_resamp = int(round(n_samples * (target_sr / float(sr_in))))
  # stft frames (center=True)
  T = 1 + int(math.floor((n_resamp + hop//2) / hop))
  return T


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
  xs = torch.stack([b["x"] for b in batch], dim=0)         # (B,C,F,Tw)
  ys = torch.stack([b["y"] for b in batch], dim=0)         # (B,Tw)
  cnt = torch.stack([b["count"] for b in batch], dim=0)    # (B,)
  return {"x": xs, "y": ys, "count": cnt}
