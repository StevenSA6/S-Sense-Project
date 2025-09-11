from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, OrderedDict, Tuple, Any
from pathlib import Path
import librosa
import soundfile as sf
import csv
import numpy as np

import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig
from .preprocess import load_audio, preprocess_waveform
from .features import FeatCfg, extract_features
from dataio.utils import WindowCfg, window_into_chunks
from .augment import augment_waveform, specaugment
from dataio.targets import events_to_frame_targets, counts_from_events


@dataclass
class DatasetConfig:
  cfg: DictConfig
  train: bool


def load_events_csv(csv_path: Path) -> List[Dict[str, float]]:
  """Read timestamps CSV into a list of events with onset/offset."""
  if not csv_path.exists():
    return []
  events = []
  with csv_path.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      s = float(row["start"])
      e = float(row.get("end", s))
      if e < s:
        e = s
      events.append({"onset": s, "offset": e})
  return events


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
        type=getattr(fcfg, "type", "mel"),
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
        aux_env_rms=fcfg.aux_channels.envelope_rms and getattr(
            self.cfg.preprocess, "enabled", True),
        standardize=fcfg.standardize.enabled,
        standardize_per_recording=fcfg.standardize.per_recording,
        calibration_secs=fcfg.standardize.calibration_secs,
    )
    self.win_cfg = WindowCfg(
        chunk_sec=self.cfg.windowing.chunk_sec,
        overlap=self.cfg.windowing.overlap,
        pad_mode=self.cfg.windowing.pad_mode,
    )

    win, hop = _frame_params_ms(
        self.feat_cfg.sr, self.feat_cfg.win_ms, self.feat_cfg.hop_ms)
    self.frames_per_chunk = int(
        round(self.win_cfg.chunk_sec * self.feat_cfg.sr / hop))
    self.hop = hop
    self.index: List[Tuple[int, int]] = []

    for i, it in enumerate(self.items):
      n_frames = _estimate_frames(
          it["path"], target_sr=self.feat_cfg.sr, win=win, hop=hop)
      step = max(1, int(round(self.frames_per_chunk *
                              (1.0 - self.win_cfg.overlap))))
      if n_frames < self.frames_per_chunk:
        n_windows = 1
      else:
        n_windows = 1 + math.floor(
            (n_frames - self.frames_per_chunk) / step)
      for w in range(n_windows):
        self.index.append((i, w))
    self._cache = OrderedDict()  # path -> (X, hop, y_frames, events)
    self._cache_cap = int(getattr(self.cfg.data, "max_cached_files", 8))

  def _get_preprocessed(self, path: Path, events_hint):
    key = str(path)
    if key in self._cache:               # hit
      X, hop, y_frames, events = self._cache.pop(key)
      self._cache[key] = (X, hop, y_frames, events)
      return X, hop, y_frames, events

    # --- compute ONCE per file ---
    y, sr_raw = load_audio(str(path), expected_sr=None, mono=True)
    y_p, aux = preprocess_waveform(y, sr_raw, self.cfg)     # per-file DSP

    X, _, hop = extract_features(
        y_p, self.cfg.audio_io.model_sr, self.feat_cfg, aux=aux)

    # labels (load if not supplied in manifest)
    events = events_hint
    if events is None:
      audio_root = Path(self.cfg.paths.audio_dir)
      ts_root = Path(self.cfg.paths.timestamps_dir)
      rel = path.relative_to(audio_root).with_suffix(".csv")
      events = load_events_csv(ts_root / rel)

    weak = float(self.cfg.labels.get("weak_dur_sec", 0.4))
    soft = int(self.cfg.labels.get("positive_dilation_frames", 0))
    minf = int(round(self.cfg.labels.min_event_sec * self.cfg.audio_io.model_sr / hop)) \
        if "min_event_sec" in self.cfg.labels else 0

    y_frames = events_to_frame_targets(
        events, T=X.shape[-1], sr=self.cfg.audio_io.model_sr, hop=hop,
        weak_dur_s=weak, soft_dilate_frames=soft, min_event_frames=minf
    ).astype(np.float32)

    # LRU insert
    self._cache[key] = (X, hop, y_frames, events)
    if len(self._cache) > self._cache_cap:
      self._cache.popitem(last=False)  # evict oldest
    return X, hop, y_frames, events

  def __len__(self) -> int:
    return len(self.index)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    file_idx, w_ord = self.index[idx]
    item = self.items[file_idx]
    path = Path(item["path"])

    # compute once per recording, reuse thereafter
    X, hop, y_full, events = self._get_preprocessed(path, item.get("events"))

    W, idxs = window_into_chunks(
        X, self.cfg.audio_io.model_sr, hop, self.win_cfg)
    if not idxs:
      W = X[:, :, :self.frames_per_chunk][None, ...]
      idxs = [(0, min(self.frames_per_chunk, X.shape[-1]))]
    w_ord = min(w_ord, len(idxs)-1)
    x = W[w_ord]
    t0, t1 = idxs[w_ord]

    if self.train and self.cfg.augment.enabled and random.random() < self.cfg.augment.prob.specaugment:
      x0 = specaugment(
          x[0], self.cfg.audio_io.model_sr, hop,
          self.cfg.augment.specaugment.time_mask_ms,
          self.cfg.augment.specaugment.time_masks,
          self.cfg.augment.specaugment.freq_mask_bins,
          self.cfg.augment.specaugment.freq_masks,
      )
      x = np.concatenate([x0[None, ...], x[1:]], axis=0)

    y_win = y_full[t0:t1]
    count_win = counts_from_events(
        events, [(t0, t1)], sr=self.cfg.audio_io.model_sr, hop=hop)[0]

    return {"x": torch.from_numpy(x),
            "y": torch.from_numpy(y_win).float(),
            "count": torch.tensor(float(count_win), dtype=torch.float32)}


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
  xs = torch.stack([b["x"] for b in batch], dim=0)
  ys = torch.stack([b["y"] for b in batch], dim=0)
  cnt = torch.stack([b["count"] for b in batch], dim=0)
  return {"x": xs, "y": ys, "count": cnt}


def _frame_params_ms(sr: int, win_ms: float, hop_ms: float) -> Tuple[int, int]:
  win = int(round(sr * win_ms / 1000.0))
  hop = int(round(sr * hop_ms / 1000.0))
  return win, hop


def _estimate_frames(path: str, target_sr: int, win: int, hop: int) -> int:
  """Estimate frame count after resampling to target_sr without loading audio."""
  try:
    info = sf.info(path)
    duration_s = info.frames / float(info.samplerate)
  except Exception:
    # librosa will use audioread as needed; use new kwarg 'path'
    duration_s = float(librosa.get_duration(path=path))
  n_resamp = int(round(duration_s * target_sr))
  T = 1 + int(math.floor((n_resamp + hop // 2) / hop))
  return T
