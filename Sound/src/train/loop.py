from __future__ import annotations
import os
import re
import time
from typing import Any, Dict, Tuple, cast

import torch
from torch.utils.data import DataLoader
# fmt: off
from torch.utils.tensorboard import SummaryWriter # pyright: ignore[reportPrivateImportUsage]
# fmt: on
from omegaconf import DictConfig, OmegaConf

from dataio.dataset import SwallowWindowDataset, DatasetConfig, collate_batch
from dataio.discovery import scan_audio_dir
from dataio.splits import make_folds
from models.sed_crnn import CRNN
from models.losses import sed_loss, count_losses

# ---------- util ----------


def _make_run_dir(cfg: DictConfig, suffix: str = "") -> str:
  ts = time.strftime("%Y%m%d-%H%M%S")
  name = (cfg.logging.run_naming.replace("{project_name}", cfg.project_name)
          .replace("{run_name}", f"{cfg.run_name}{suffix}")
          .replace("{time}", ts))
  run_dir = os.path.join(cfg.paths.out_dir, "runs", name)
  os.makedirs(run_dir, exist_ok=True)
  OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
  return run_dir


def _metric_selection(cfg: DictConfig, val_metrics: Dict[str, float]) -> float:
  key = cfg.logging.checkpoints.monitor
  return float(val_metrics.get(key, val_metrics.get("val/frame_f1", -val_metrics.get("val/loss", 0.0))))


def _is_better(a: float, b: float, mode: str) -> bool:
  return (a > b) if mode == "max" else (a < b)


def _prune_topk(run_dir: str, top_k: int):
  files = [f for f in os.listdir(run_dir) if f.startswith(
      "epoch_") and f.endswith(".ckpt")]
  scored = []
  for f in files:
    m = re.search(r"_(?:mon|val)[^_]*_([-+]?[0-9]*\.?[0-9]+)\.ckpt$", f)
    if m:
      try:
        scored.append((float(m.group(1)), f))
      except ValueError:
        pass
  scored.sort(key=lambda x: x[0], reverse=True)
  for _, f in scored[top_k:]:
    try:
      os.remove(os.path.join(run_dir, f))
    except OSError:
      pass

# ---------- core ----------


def train_one_fold(cfg: DictConfig, fold_id: int):
  device = cfg.hardware.device

  # discover files and subjects
  items = scan_audio_dir(cfg.paths.audio_dir)
  folds = make_folds(items, cfg.cv.folds, cfg.cv.split_by)
  train_idx, val_idx = folds[fold_id]
  train_items = [items[i] for i in train_idx]
  val_items = [items[i] for i in val_idx]

  # datasets/dataloaders
  train_ds = SwallowWindowDataset(
      train_items, DatasetConfig(cfg=cfg, train=True))
  val_ds = SwallowWindowDataset(
      val_items,   DatasetConfig(cfg=cfg, train=False))
  train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        num_workers=cfg.hardware.num_workers, pin_memory=True, collate_fn=collate_batch)
  val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False,
                      num_workers=cfg.hardware.num_workers, pin_memory=True, collate_fn=collate_batch)

  # model + optim
  net = CRNN(cast(Dict[str, Any], OmegaConf.to_container(
      cfg, resolve=True))).to(device)
  opt = torch.optim.AdamW(net.parameters(), lr=cfg.optimizer.lr,  # pyright: ignore[reportPrivateImportUsage]
                          weight_decay=cfg.optimizer.weight_decay, betas=tuple(cfg.optimizer.betas))
  sch = None
  if getattr(cfg, "scheduler", None) and cfg.scheduler.type == "cosine":
    T, warm = cfg.training.epochs, cfg.scheduler.warmup_epochs

    def lr_lambda(e):
      if e < warm:
        return (e+1)/max(1, warm)
      prog = (e - warm) / max(1, T - warm)
      return 0.5*(1.0 + torch.cos(torch.tensor(prog*3.1415926535)))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: float(lr_lambda(e)))

  # bookkeeping
  run_dir = _make_run_dir(cfg, suffix=f"_fold{fold_id}")
  writer = SummaryWriter(os.path.join(run_dir, "tb")
                         ) if cfg.logging.tensorboard else None
  wandb_run = None
  if cfg.logging.wandb.enabled:
    import wandb  # pyright: ignore[reportMissingImports]
    wandb_run = wandb.init(project=cfg.project_name,
                           entity=cfg.logging.wandb.entity,
                           name=f"{cfg.run_name}_fold{fold_id}",
                           dir=run_dir,
                           config=OmegaConf.to_container(cfg, resolve=True))
  scaler = torch.cuda.amp.GradScaler(enabled=(cfg.hardware.precision == 16))
  best = float(
      "-inf") if cfg.logging.checkpoints.mode == "max" else float("inf")
  patience = cfg.training.early_stop.patience if cfg.training.early_stop.enabled else 10
  bad = 0
  global_step = 0

  # epochs
  for epoch in range(cfg.training.epochs):
    net.train()
    for b in train_dl:
      xb, yb, cb = b["x"].to(device), b["y"].to(device), b["count"].to(device)
      opt.zero_grad(set_to_none=True)
      with torch.autocast("cuda", enabled=(cfg.hardware.precision == 16)):
        out = net(xb)
        logits = out["logits"]
        assert isinstance(
            cfg, Dict), f"cfg should be of type Dict, but got type {type(cfg)} instead"
        loss = sed_loss(logits, yb, cfg.loss.sed) + \
            count_losses(out, cb, logits, train_ds.hop,
                         cfg.audio_io.model_sr, cfg)
      scaler.scale(loss).backward()
      torch.nn.utils.clip_grad_norm_(
          net.parameters(), cfg.optimizer.grad_clip_norm)
      scaler.step(opt)
      scaler.update()
      if writer:
        writer.add_scalar("train/loss", float(loss.item()), global_step)
      if wandb_run:
        wandb_run.log({"train/loss": float(loss.item())}, step=global_step)
      global_step += 1
    if sch is not None:
      sch.step()

    # validate
    net.eval()
    tp = fp = fn = 0
    val_loss = 0.0
    with torch.no_grad():
      for b in val_dl:
        xb, yb = b["x"].to(device), b["y"].to(device)
        logits = net(xb)["logits"]
        val_loss += float(sed_loss(logits, yb,
                          cfg.loss.sed).detach().cpu().item())
        pred = (torch.sigmoid(logits) > 0.5)
        tp += (pred & (yb > 0.5)).sum().item()
        fp += (pred & (yb <= 0.5)).sum().item()
        fn += ((~pred) & (yb > 0.5)).sum().item()
    prec = tp/(tp+fp+1e-9)
    rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    val_loss /= max(1, len(val_dl))
    val_metrics = {"val/loss": float(val_loss),
                   "val/frame_precision": float(prec),
                   "val/frame_recall": float(rec),
                   "val/frame_f1": float(f1)}

    if writer:
      for k, v in val_metrics.items():
        writer.add_scalar(k, v, epoch)
    if wandb_run:
      wandb_run.log(val_metrics, step=epoch)

    mon = _metric_selection(cfg, val_metrics)
    key = cfg.logging.checkpoints.monitor.replace("/", "-")
    ck = os.path.join(run_dir, f"epoch_{epoch:03d}_{key}_{mon:.4f}.ckpt")
    state = {"epoch": epoch, "model_state": net.state_dict(
    ), "optimizer_state": opt.state_dict(), "metrics": val_metrics}
    if cfg.logging.save_every_epoch:
      torch.save(state, ck)
    if cfg.logging.save_last:
      torch.save(state, os.path.join(run_dir, "last.ckpt"))

    if cfg.logging.save_best and _is_better(mon, best, cfg.logging.checkpoints.mode):
      best = mon
      torch.save(state, os.path.join(run_dir, "best.ckpt"))
      if int(cfg.logging.checkpoints.save_top_k) > 0:
        _prune_topk(run_dir, top_k=int(cfg.logging.checkpoints.save_top_k))
      bad = 0
    else:
      bad += 1
      if cfg.training.early_stop.enabled and bad >= patience:
        break

    if writer:
      writer.close()
    if wandb_run:
      wandb_run.finish()
    # load best for evaluation caller
    best_ckpt = os.path.join(run_dir, "best.ckpt")
    # keep the in-memory model already at last epoch; optionally reload best:
    if os.path.exists(best_ckpt):
      state = torch.load(best_ckpt, map_location=device)
      net.load_state_dict(state["model_state"])
    return best_ckpt, net
