from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from scipy.signal import medfilt

from omegaconf import DictConfig
from dataio.preprocess import load_audio, preprocess_waveform
from dataio.features import FeatCfg, WindowCfg, extract_features, window_into_chunks


def smooth_posteriors(p: np.ndarray, k: int, mode: str) -> np.ndarray:
  if k <= 1 or mode == "none":
    return p
  if mode == "moving":
    k = int(k) | 1  # odd
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(p, w, mode="same").astype(np.float32)
  if mode == "median":
    k = int(k) | 1
    return medfilt(p, kernel_size=k).astype(np.float32)
  return p


def hysteresis_decode(p: np.ndarray, th_on: float, th_off: float) -> List[Tuple[int, int]]:
  on = p > th_on
  off = p < th_off
  segs: List[Tuple[int, int]] = []
  i = 0
  T = len(p)
  while i < T:
    while i < T and not on[i]:
      i += 1
    if i >= T:
      break
    s = i
    i += 1
    while i < T and not off[i]:
      i += 1
    e = i
    segs.append((s, e))
  return segs


def enforce_minima(
    segs: List[Tuple[int, int]],
    min_dur: int,
    min_gap: int
) -> List[Tuple[int, int]]:
  if not segs:
    return segs
  # drop too short
  segs = [(s, e) for (s, e) in segs if (e - s) >= min_dur]
  if not segs:
    return segs
  # merge gaps < min_gap
  merged: List[Tuple[int, int]] = [segs[0]]
  for s, e in segs[1:]:
    ps, pe = merged[-1]
    if s - pe < min_gap:
      merged[-1] = (ps, max(pe, e))
    else:
      merged.append((s, e))
  return merged


def viterbi_two_state(p: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
  """
  p = P(event) per frame. States: 0=silence, 1=event
  log_trans: 2x2 log transition matrix [[ll00,ll01],[ll10,ll11]]
  """
  T = len(p)
  log_p = np.stack([np.log1p(-p + 1e-12), np.log(p + 1e-12)], axis=1)  # (T,2)
  dp = np.full((T, 2), -1e30, dtype=np.float32)
  ptr = np.zeros((T, 2), dtype=np.int8)
  dp[0] = log_p[0]
  for t in range(1, T):
    for s in (0, 1):
      cand = dp[t-1] + log_trans[:, s]
      ptr[t, s] = np.argmax(cand)
      dp[t, s] = cand[ptr[t, s]] + log_p[t, s]
  states = np.zeros(T, dtype=np.int8)
  states[-1] = np.argmax(dp[-1])
  for t in range(T-2, -1, -1):
    states[t] = ptr[t+1, states[t+1]]
  return states  # 0/1 per frame


def states_to_segments(states: np.ndarray) -> List[Tuple[int, int]]:
  segs: List[Tuple[int, int]] = []
  T = len(states)
  i = 0
  while i < T:
    while i < T and states[i] == 0:
      i += 1
    if i >= T:
      break
    s = i
    while i < T and states[i] == 1:
      i += 1
    e = i
    segs.append((s, e))
  return segs


def stitch_over_windows(
    posteriors_windows: List[np.ndarray],  # each (Tw,)
    idxs: List[Tuple[int, int]],
    T_total: int
) -> np.ndarray:
  acc = np.zeros(T_total, dtype=np.float32)
  cnt = np.zeros(T_total, dtype=np.float32)
  for p, (t0, t1) in zip(posteriors_windows, idxs):
    t1c = min(t1, T_total)
    acc[t0:t1c] += p[:t1c-t0]
    cnt[t0:t1c] += 1.0
  cnt[cnt == 0.0] = 1.0
  return (acc / cnt).astype(np.float32)


@torch.no_grad()
def infer_path(
    model: torch.nn.Module,
    cfg: DictConfig,
    path: str,
    batch_size: int = 16,
) -> Dict:
  if getattr(cfg, "baseline", None) and getattr(cfg.baseline, "enabled", False):
    from baseline.detector import detect_path
    return detect_path(path, cfg)
  model.eval()
  # load + DSP
  y, sr_raw = load_audio(path, expected_sr=None, mono=True)
  y_p, aux = preprocess_waveform(y, sr_raw, cfg)

  # features (full length)
  fcfg = cfg.features
  feat_cfg = FeatCfg(
      sr=cfg.audio_io.model_sr,
      n_fft=getattr(fcfg, "n_fft", 512),
      win_ms=fcfg.win_ms, hop_ms=fcfg.hop_ms,
      n_mels=fcfg.n_mels, fmin=fcfg.fmin, fmax=fcfg.fmax,
      log_eps=getattr(fcfg, "log_eps", 1e-6),
      deltas_enabled=fcfg.deltas.enabled, deltas_order=fcfg.deltas.order,
      aux_enabled=False,  # no aux at inference unless you want it tiled
  )
  X, _, hop = extract_features(y_p, cfg.audio_io.model_sr, feat_cfg, aux=None)
  T_full = X.shape[-1]

  # windowing
  wcfg = WindowCfg(chunk_sec=cfg.windowing.chunk_sec,
                   overlap=cfg.windowing.overlap,
                   pad_mode=cfg.windowing.pad_mode)
  W, idxs = window_into_chunks(X, cfg.audio_io.model_sr, hop, wcfg)
  if len(W) == 0:  # degenerate
    W = X[:, :, :1][None, ...]
    idxs = [(0, 1)]

  # forward in batches
  device = cfg.hardware.device
  outs: List[np.ndarray] = []
  for i in range(0, len(W), batch_size):
    xb = torch.from_numpy(W[i:i+batch_size]).to(device)
    out = model(xb)["logits"]         # (B,Tw)
    pb = torch.sigmoid(out).cpu().numpy()
    outs.extend([p for p in pb])

  # stitch
  p_full = stitch_over_windows(outs, idxs, T_total=T_full)

  # optional HMM
  if cfg.inference.get("crf", {}).get("enabled", False):
    logA = np.array(cfg.inference.crf.transition,
                    dtype=np.float32)  # 2x2 in log-space
    states = viterbi_two_state(p_full, logA)
    segs = states_to_segments(states)
  else:
    # smooth + hysteresis
    k = int(cfg.inference.smoothing.k_frames)
    mode = cfg.inference.smoothing.type
    p_sm = smooth_posteriors(p_full, k=k, mode=mode)
    segs = hysteresis_decode(
        p_sm, cfg.inference.hysteresis.th_on, cfg.inference.hysteresis.th_off)

  # constraints
  min_dur = int(round(cfg.inference.constraints.min_event_sec *
                cfg.audio_io.model_sr / hop))
  min_gap = int(round(cfg.inference.constraints.min_gap_sec *
                cfg.audio_io.model_sr / hop))
  segs = enforce_minima(segs, min_dur=min_dur, min_gap=min_gap)

  # to seconds
  onsets = [s * hop / cfg.audio_io.model_sr for (s, _) in segs]
  offsets = [e * hop / cfg.audio_io.model_sr for (_, e) in segs]

  return {
      "posteriors": p_full,          # (T,)
      "segments_frames": segs,       # [(s,e)]
      "onsets_s": onsets,
      "offsets_s": offsets,
      "count": len(segs),
      "hop": hop,
      "sr": cfg.audio_io.model_sr,
  }
