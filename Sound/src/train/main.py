import os
import re
import time
from typing import Any, Dict, cast

import torch
from torch.utils.data import DataLoader
# fmt: off
from torch.utils.tensorboard import SummaryWriter # pyright: ignore[reportPrivateImportUsage]
# fmt: on
from omegaconf import DictConfig, OmegaConf
import wandb

from src.dataio.dataset import (
    SwallowWindowDataset,
    DatasetConfig,
    read_jsonl,
    collate_batch,
)
from src.models.sed_crnn import CRNN
from src.models.losses import sed_loss, count_losses


def _make_run_dir(cfg) -> str:
  ts = time.strftime("%Y%m%d-%H%M%S")
  name = (
      cfg.logging.run_naming.replace("{project_name}", cfg.project_name)
      .replace("{run_name}", cfg.run_name)
      .replace("{time}", ts)
  )
  run_dir = os.path.join(cfg.paths.out_dir, "runs", name)
  os.makedirs(run_dir, exist_ok=True)
  OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
  return run_dir


def _build_loaders(cfg):
  train_items = read_jsonl(cfg.paths.train_manifest)
  val_items = read_jsonl(cfg.paths.val_manifest)

  train_ds = SwallowWindowDataset(
      train_items, DatasetConfig(cfg=cfg, train=True))
  val_ds = SwallowWindowDataset(val_items, DatasetConfig(cfg=cfg, train=False))

  train_dl = DataLoader(
      train_ds,
      batch_size=cfg.training.batch_size,
      shuffle=True,
      num_workers=cfg.hardware.num_workers,
      pin_memory=True,
      collate_fn=collate_batch,
  )
  val_dl = DataLoader(
      val_ds,
      batch_size=cfg.training.batch_size,
      shuffle=False,
      num_workers=cfg.hardware.num_workers,
      pin_memory=True,
      collate_fn=collate_batch,
  )
  return train_dl, val_dl, train_ds


def _make_model_and_optim(cfg):
  device = cfg.hardware.device
  raw_container = OmegaConf.to_container(cfg, resolve=True)
  assert isinstance(raw_container, Dict)
  net = CRNN(raw_container).to(device)

  opt = torch.optim.AdamW(  # pyright: ignore[reportPrivateImportUsage]
      net.parameters(),
      lr=cfg.optimizer.lr,
      weight_decay=cfg.optimizer.weight_decay,
      betas=tuple(cfg.optimizer.betas),
  )

  # Cosine schedule with warmup if requested
  if getattr(cfg, "scheduler", None) and cfg.scheduler.type == "cosine":
    T = cfg.training.epochs
    warm = cfg.scheduler.warmup_epochs

    def lr_lambda(e):
      if e < warm:
        return (e + 1) / max(1, warm)
      prog = (e - warm) / max(1, T - warm)
      return 0.5 * (1.0 + torch.cos(torch.tensor(prog * 3.1415926535)))

    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: float(lr_lambda(e)))
  else:
    sch = None

  return net, opt, sch


def _metric_selection(cfg, val_metrics: Dict[str, float]) -> float:
  key = cfg.logging.checkpoints.monitor
  mode = cfg.logging.checkpoints.mode  # "max" or "min"
  # Fallbacks if the configured key isn’t logged this epoch
  monitored = val_metrics.get(
      key, val_metrics.get(
          "val/event_f1", val_metrics.get("val/frame_f1", None))
  )
  if monitored is None:
    # last resort: negative loss so "max" favors lower loss
    monitored = -val_metrics.get("val/loss", 0.0)
  return float(monitored)


def _is_better(a: float, b: float, mode: str) -> bool:
  return (a > b) if mode == "max" else (a < b)


def _save_ckpt(path: str, net, opt, epoch: int, metrics: Dict[str, float]):
  state = {
      "epoch": epoch,
      "model_state": net.state_dict(),
      "optimizer_state": opt.state_dict(),
      "metrics": metrics,
  }
  torch.save(state, path)


def _prune_topk(run_dir: str, top_k: int):
  files = [
      f for f in os.listdir(run_dir) if f.startswith("epoch_") and f.endswith(".ckpt")
  ]
  scored = []
  for f in files:
    m = re.search(r"_(?:mon|val)[^_]*_([-+]?[0-9]*\.?[0-9]+)\.ckpt$", f)
    if m:
      try:
        scored.append((float(m.group(1)), f))
      except ValueError:
        pass
  if not scored:
    return
  # Keep highest scores first; filenames encode the monitored metric value
  scored.sort(key=lambda x: x[0], reverse=True)
  for _, f in scored[top_k:]:
    try:
      os.remove(os.path.join(run_dir, f))
    except OSError:
      pass


def main():
  cfg = OmegaConf.load("configs/initial_config.yaml")
  device = cfg.hardware.device

  run_dir = _make_run_dir(cfg)
  writer = SummaryWriter(os.path.join(run_dir, "tb")
                         ) if cfg.logging.tensorboard else None

  use_wandb = getattr(cfg.logging, "wandb", {}).get("enabled", False)
  if use_wandb:
    raw_container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw_container, Dict) and raw_container is not None
    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name,
        entity=cfg.logging.wandb.entity,
        config=cast(Dict[str, Any], raw_container),
    )

  train_dl, val_dl, train_ds = _build_loaders(cfg)
  net, opt, sch = _make_model_and_optim(cfg)

  scaler = torch.cuda.amp.GradScaler(enabled=(cfg.hardware.precision == 16))
  best = float(
      "-inf") if cfg.logging.checkpoints.mode == "max" else float("inf")
  patience = cfg.training.early_stop.patience if cfg.training.early_stop.enabled else 10
  bad = 0
  global_step = 0

  for epoch in range(cfg.training.epochs):
    # -------- train --------
    net.train()
    for b in train_dl:
      xb = b["x"].to(device)         # (B,C,F,Tw)
      yb = b["y"].to(device)         # (B,Tw)
      cb = b["count"].to(device)     # (B,)

      opt.zero_grad(set_to_none=True)
      with torch.autocast("cuda", enabled=(cfg.hardware.precision == 16)):
        out = net(xb)
        logits = out["logits"]
        assert isinstance(
            cfg, dict), f"cfg should be of type Dict but got type f{type(cfg)} instead"
        loss = sed_loss(logits, yb, cfg.loss.sed) + count_losses(
            out, cb, logits, train_ds.hop, cfg.audio_io.model_sr, cfg
        )
      scaler.scale(loss).backward()
      torch.nn.utils.clip_grad_norm_(
          net.parameters(), cfg.optimizer.grad_clip_norm)
      scaler.step(opt)
      scaler.update()

      if writer:
        writer.add_scalar("train/loss", float(loss.item()), global_step)
      if use_wandb:
        wandb.log({"train/loss": float(loss.item()),
                  "step": global_step, "epoch": epoch})
      global_step += 1

    if sch is not None:
      sch.step()

    # -------- validate (frame metrics) --------
    net.eval()
    tp = fp = fn = 0
    val_loss = 0.0
    with torch.no_grad():
      for b in val_dl:
        xb = b["x"].to(device)
        yb = b["y"].to(device)
        out = net(xb)
        logits = out["logits"]
        val_loss += float(
            (sed_loss(logits, yb, cfg.loss.sed)).detach().cpu().item()
        )
        pred = (torch.sigmoid(logits) > 0.5)
        tp += (pred & (yb > 0.5)).sum().item()
        fp += (pred & (yb <= 0.5)).sum().item()
        fn += ((~pred) & (yb > 0.5)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    val_loss /= max(1, len(val_dl))

    val_metrics = {
        "val/loss": float(val_loss),
        "val/frame_precision": float(prec),
        "val/frame_recall": float(rec),
        "val/frame_f1": float(f1),
        # Add "val/event_f1" later when you wire full-file evaluation
    }

    if writer:
      for k, v in val_metrics.items():
        writer.add_scalar(k, v, epoch)
    if use_wandb:
      wandb.log({**val_metrics, "epoch": epoch})

    # -------- checkpoints --------
    monitor_value = _metric_selection(cfg, val_metrics)
    monitor_key = cfg.logging.checkpoints.monitor.replace("/", "-")
    mode = cfg.logging.checkpoints.mode
    save_top_k = int(cfg.logging.checkpoints.save_top_k)

    state_name = f"epoch_{epoch:03d}_{monitor_key}_{monitor_value:.4f}.ckpt"

    if cfg.logging.save_every_epoch:
      _save_ckpt(os.path.join(run_dir, state_name),
                 net, opt, epoch, val_metrics)

    if cfg.logging.save_last:
      _save_ckpt(os.path.join(run_dir, "last.ckpt"),
                 net, opt, epoch, val_metrics)

    if cfg.logging.save_best and _is_better(monitor_value, best, mode):
      best = monitor_value
      _save_ckpt(os.path.join(run_dir, "best.ckpt"),
                 net, opt, epoch, val_metrics)
      if save_top_k > 0:
        _prune_topk(run_dir, top_k=save_top_k)
      bad = 0
    else:
      bad += 1
      if cfg.training.early_stop.enabled and bad >= patience:
        break

  if writer:
    writer.close()
  if use_wandb:
    wandb.finish()


if __name__ == "__main__":
  main()
