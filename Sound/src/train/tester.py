# src/train/tester.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from omegaconf import DictConfig
from dataio.discovery import scan_audio_dir
from dataio.splits import make_folds
from infer.inference import infer_path
from infer.eval_metrics import event_f1_sed_eval, count_mae_mape, onset_mae


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


def _refs_for(items: List[Dict[str, Any]], timestamps_dir: str) -> Dict[str, List[Dict[str, float]]]:
  refs: Dict[str, List[Dict[str, float]]] = {}
  for it in items:
    wav = Path(it["path"])
    refs[str(wav)] = _load_events_csv(Path(timestamps_dir) / f"{wav.stem}.csv")
  return refs


@torch.no_grad()
def evaluate_fold(cfg: DictConfig, fold_id: int, model: torch.nn.Module) -> Dict[str, float]:
  # validation set = held-out subjects of this fold
  items = scan_audio_dir(cfg.paths.audio_dir)
  train_idx, val_idx = make_folds(
      items, cfg.cv.folds, cfg.cv.split_by)[fold_id]
  val_items = [items[i] for i in val_idx]

  # run inference per file
  preds_by_file: Dict[str, Tuple[List[float], List[float]]] = {}
  for it in val_items:
    r = infer_path(model, cfg, it["path"])
    preds_by_file[it["path"]] = (r["onsets_s"], r["offsets_s"])

  # refs
  refs_by_file = _refs_for(val_items, cfg.paths.timestamps_dir)

  # metrics
  ev = event_f1_sed_eval(refs_by_file, preds_by_file,
                         onset_tol=float(cfg.metrics.onset_tol_sec),
                         offset_tol=float(cfg.metrics.offset_tol_sec))
  cnt = count_mae_mape(refs_by_file, preds_by_file)
  omt = onset_mae(refs_by_file, preds_by_file,
                  match_tol=float(cfg.metrics.onset_tol_sec))

  out = {"event_f1": ev["F1"], "event_precision": ev["Precision"],
         "event_recall": ev["Recall"], "count_mae": cnt["MAE"],
         "count_mape_%": cnt["MAPE_%"], "onset_mae_sec": omt}
  return out
