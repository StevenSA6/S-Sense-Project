# eval_metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import sed_eval
from collections import defaultdict


def _to_sed_events(events: List[Dict[str, float]]) -> List[Dict[str, Any]]:
  out = []
  for e in events:
    onset = float(e["onset"])
    offset = float(e.get("offset", onset))
    out.append({"event_label": "swallow",
               "event_onset": onset, "event_offset": offset})
  return out


def _pred_to_sed(onsets: List[float], offsets: List[float]) -> List[Dict[str, Any]]:
  return [{"event_label": "swallow", "event_onset": float(o), "event_offset": float(f)} for o, f in zip(onsets, offsets)]


def event_f1_sed_eval(
    refs_by_file: Dict[str, List[Dict[str, float]]],
    preds_by_file: Dict[str, Tuple[List[float], List[float]]],
    onset_tol: float = 0.2,
    offset_tol: Optional[float] = 0.2,
) -> Dict[str, float]:
  """Returns {'F1':..., 'Precision':..., 'Recall':...} micro-averaged over files."""
  ebm = sed_eval.sound_event.EventBasedMetrics(
      event_label_list=["swallow"],
      t_collar=onset_tol,
      percentage_of_length=0.5,
      empty_system_output_handling="zero_score",
  ) if offset_tol is not None else \
      sed_eval.sound_event.EventBasedMetrics(
      event_label_list=["swallow"],
      t_collar=onset_tol,
      percentage_of_length=0.5,
      empty_system_output_handling="zero_score",
  )
  for path, ref_events in refs_by_file.items():
    gt = _to_sed_events(ref_events)
    ons, offs = preds_by_file.get(path, ([], []))
    pd = _pred_to_sed(ons, offs)
    ebm.evaluate(reference_event_list=gt, estimated_event_list=pd)
  f = ebm.results_overall_metrics()
  return {"F1": f["f_measure"]["f_measure"], "Precision": f["f_measure"]["precision"], "Recall": f["f_measure"]["recall"]}


def count_mae_mape(
    refs_by_file: Dict[str, List[Dict[str, float]]],
    preds_by_file: Dict[str, Tuple[List[float], List[float]]],
) -> Dict[str, float]:
  y_true, y_pred = [], []
  for path, ref_events in refs_by_file.items():
    y_true.append(len(ref_events))
    ons, offs = preds_by_file.get(path, ([], []))
    y_pred.append(len(ons))
  y_true = np.asarray(y_true, float)
  y_pred = np.asarray(y_pred, float)
  mae = np.mean(np.abs(y_true - y_pred))
  mape = np.mean(np.abs((y_true - y_pred) /
                 np.clip(y_true, 1e-6, None))) * 100.0
  return {"MAE": float(mae), "MAPE_%": float(mape)}


def onset_mae(
    refs_by_file: Dict[str, List[Dict[str, float]]],
    preds_by_file: Dict[str, Tuple[List[float], List[float]]],
    match_tol: float = 0.2
) -> float:
  """Greedy nearest-onset matching with tolerance."""
  errs = []
  for path, ref_events in refs_by_file.items():
    r = [e["onset"] for e in ref_events]
    p = list(preds_by_file.get(path, ([], []))[0])
    if not r or not p:
      continue
    used = np.zeros(len(p), dtype=bool)
    for rr in r:
      j = int(np.argmin([abs(rr - pp) for pp in p])) if p else None
      if j is None:
        continue
      if used[j]:
        continue
      err = abs(rr - p[j])
      if err <= match_tol:
        errs.append(err)
        used[j] = True
  return float(np.mean(errs)) if errs else float("nan")


def subject_macro(
    refs_by_file: Dict[str, List[Dict[str, float]]],
    preds_by_file: Dict[str, Tuple[List[float], List[float]]],
    file_to_subject: Dict[str, str],
    onset_tol: float = 0.2
) -> Dict[str, float]:
  """Macro-average F1 across subjects."""
  by_sub_refs = defaultdict(dict)
  by_sub_preds = defaultdict(dict)
  for path in refs_by_file:
    s = file_to_subject.get(path, "UNK")
    by_sub_refs[s][path] = refs_by_file[path]
    by_sub_preds[s][path] = preds_by_file.get(path, ([], []))
  scores = []
  for s in by_sub_refs:
    f1 = event_f1_sed_eval(
        by_sub_refs[s], by_sub_preds[s], onset_tol=onset_tol)["F1"]
    scores.append(f1)
  return {"SubjectMacroF1": float(np.mean(scores)) if scores else float("nan")}
