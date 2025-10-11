# src/train/tester.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from omegaconf import DictConfig
from dataio.utils import scan_audio_dir
from train.utils import make_folds
from infer.inference import infer_path
from infer.eval_metrics import event_f1_sed_eval, count_mae_mape, onset_mae
import numpy as np
import pandas as pd


def _load_events_csv(csv_path: Path) -> List[Dict[str, float]]:
  import csv
  if not csv_path.exists():
    return []
  ev = []
  with csv_path.open("r", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
      s = float(r["start"])
      e = float(r.get("end", s))
      if e < s:
        e = s
      ev.append({"onset": s, "offset": e})
  return ev


def _refs_for(items: List[Dict[str, Any]],
              audio_dir: str,
              timestamps_dir: str) -> Dict[str, List[Dict[str, float]]]:
  refs: Dict[str, List[Dict[str, float]]] = {}
  audio_root = Path(audio_dir)
  audio_root_abs = audio_root.resolve()
  ts_root = Path(timestamps_dir)
  for it in items:
    wav = Path(it["path"])
    key = str(wav)

    rel: Path
    if wav.is_absolute():
      try:
        rel = wav.resolve().relative_to(audio_root_abs)
      except ValueError:
        rel = Path(wav.name)
    else:
      try:
        rel = wav.relative_to(audio_root)
      except ValueError:
        rel = Path(wav.name)
    rel = rel.with_suffix(".csv")
    events = it.get("events")
    if events is None:
      events = _load_events_csv(ts_root / rel)
    refs[key] = events
  return refs


@torch.no_grad()
def evaluate_fold(cfg: DictConfig, fold_id: int, model: torch.nn.Module) -> Dict[str, float]:
  # validation set = held-out subjects of this fold
  subj_mode = getattr(getattr(cfg.data, "subject_id", {}), "mode", "prefix")
  items = scan_audio_dir(cfg.paths.audio_dir, cfg.paths.timestamps_dir)
  train_idx, val_idx = make_folds(
      items, cfg.cv.folds, cfg.cv.split_by)[fold_id]
  val_items = [items[i] for i in val_idx]

  # run inference per file
  preds_by_file: Dict[str, Tuple[List[float], List[float]]] = {}
  for it in val_items:
    r = infer_path(model, cfg, it["path"])
    preds_by_file[it["path"]] = (r["onsets_s"], r["offsets_s"])

    save_dir = Path(cfg.paths.out_dir) / "runs" / f"{cfg.run_name}_fold{fold_id}_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_stem = Path(it["path"]).stem

    # Save full inference output (small .npz per file)
    np.savez(
        save_dir / f"{file_stem}.npz",
        path=str(it["path"]),
        posteriors=r["posteriors"],
        onsets_s=np.array(r["onsets_s"]),
        offsets_s=np.array(r["offsets_s"]),
        hop=r["hop"],
        sr=r["sr"]
    )

  # refs
  refs_by_file = _refs_for(
      val_items, cfg.paths.audio_dir, cfg.paths.timestamps_dir)

  # metrics
  tol_cfg = cfg.evaluation.tolerances
  onset_tol = float(tol_cfg.onset_ms) / 1000.0
  offset_tol = float(tol_cfg.offset_ms) / 1000.0

  metrics_cfg = getattr(cfg.evaluation, "metrics", {})
  out: Dict[str, float] = {}

  if metrics_cfg.get("event_f1", True):
    ev = event_f1_sed_eval(
        refs_by_file,
        preds_by_file,
        onset_tol=onset_tol,
        offset_tol=offset_tol,
    )
    out.update({
        "event_f1": ev["F1"],
        "event_precision": ev["Precision"],
        "event_recall": ev["Recall"],
    })

  if metrics_cfg.get("count_mae", True):
    cnt = count_mae_mape(refs_by_file, preds_by_file)
    out.update({"count_mae": cnt["MAE"], "count_mape_%": cnt["MAPE_%"]})

  if metrics_cfg.get("onset_mae", True):
    omt = onset_mae(refs_by_file, preds_by_file, match_tol=onset_tol)
    out["onset_mae_sec"] = omt

  save_dir = Path(cfg.paths.out_dir) / f"predictions_fold{fold_id}"
  save_dir.mkdir(parents=True, exist_ok=True)

  file_metrics = []
  for path in preds_by_file:
    ev = event_f1_sed_eval(
        {path: refs_by_file[path]},
        {path: preds_by_file[path]},
        onset_tol=onset_tol,
        offset_tol=offset_tol,
    )
    cnt = count_mae_mape({path: refs_by_file[path]}, {
                         path: preds_by_file[path]})
    omt = onset_mae({path: refs_by_file[path]}, {
                    path: preds_by_file[path]}, match_tol=onset_tol)
    file_metrics.append({
        "file": path,
        "event_f1": ev["F1"],
        "event_precision": ev["Precision"],
        "event_recall": ev["Recall"],
        "count_mae": cnt["MAE"],
        "count_mape_%": cnt["MAPE_%"],
        "onset_mae_sec": omt,
    })

  df = pd.DataFrame(file_metrics)
  df.to_csv(save_dir / "per_file_metrics.csv", index=False)

  return out
