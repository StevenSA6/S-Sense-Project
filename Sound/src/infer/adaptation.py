from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def recalibrate_batch_norm(model: torch.nn.Module, windows: np.ndarray, device: str, batch_size: int = 16) -> None:
  """Recalibrate BatchNorm statistics with given feature windows."""
  if windows.size == 0:
    return
  model.train()
  for i in range(0, len(windows), batch_size):
    xb = torch.from_numpy(windows[i:i+batch_size]).to(device)
    model(xb)
  model.eval()


def instance_normalize(X: np.ndarray) -> np.ndarray:
  """Channel-wise instance normalization for a single recording."""
  mean = X.mean(axis=-1, keepdims=True)
  std = X.std(axis=-1, keepdims=True) + 1e-8
  return (X - mean) / std


def calibrate_posteriors(p: np.ndarray, topk: int) -> np.ndarray:
  """Simple calibration by scaling so that mean of top-k frames is 1."""
  if topk <= 0 or p.size == 0:
    return p
  k = min(len(p), int(topk))
  top_mean = np.sort(p)[-k:].mean()
  if top_mean <= 0:
    return p
  q = p / top_mean
  return np.clip(q, 0.0, 1.0)
