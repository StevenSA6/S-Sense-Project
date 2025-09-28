# stepper.py
from typing import Dict
import math
import torch
from models import build_model, infer_in_channels
from models.losses import sed_loss, count_losses


class Stepper:
  def __init__(self, cfg: Dict):
    self.cfg = cfg
    self.device = cfg["hardware"]["device"]
    in_ch = infer_in_channels(cfg)
    cfg.model.in_channels = int(in_ch)  # # type: ignore
    self.net = build_model(cfg).to(self.device)

    self.opt = torch.optim.AdamW(  # pyright: ignore[reportPrivateImportUsage]
        self.net.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"].get("betas", (0.9, 0.999))),
    )
    self.grad_clip = cfg["optimizer"].get("grad_clip_norm", 1.0)

    # cosine schedule with warmup
    T = cfg["training"]["epochs"]
    warm = cfg["scheduler"].get("warmup_epochs", 5)

    def lr_lambda(e):
      if e < warm:
        return (e + 1) / max(1, warm)
      progress = (e - warm) / max(1, T - warm)
      return 0.5 * (1.0 + math.cos(math.pi * progress))
    self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    # AMP settings
    prec = str(cfg["hardware"].get("precision", "16"))
    if prec == "bf16":
      self.use_amp, self.amp_dtype = True, torch.bfloat16
    elif prec == "16":
      self.use_amp, self.amp_dtype = True, torch.float16
    else:
      self.use_amp, self.amp_dtype = False, torch.float32

    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

  def train_epoch(self, loader, hop: int, sr: int, epoch: int) -> Dict:
    self.net.train()
    tot_loss = tot_main = tot_cnt = tot_f1 = n = 0.0
    for xb, yb, count_b in loader:
      xb, yb = xb.to(self.device), yb.to(self.device)
      count_b = count_b.to(self.device) if count_b is not None else None

      self.opt.zero_grad(set_to_none=True)
      with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
        out = self.net(xb)
        logits = out["logits"]
        loss_main = sed_loss(logits, yb, self.cfg["loss"]["sed"])
        loss_cnt = count_losses(out, count_b, logits, hop, sr, self.cfg)
        loss = loss_main + loss_cnt

      self.scaler.scale(loss).backward()
      torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
      self.scaler.step(self.opt)
      self.scaler.update()

      with torch.no_grad():
        prob = torch.sigmoid(logits)
        thr = 0.5
        f1_num = ((prob > thr) & (yb > 0.5)).sum().item() * 2
        f1_den = (prob > thr).sum().item() + (yb > 0.5).sum().item() + 1e-9
        f1 = f1_num / f1_den

      bs = xb.size(0)
      tot_loss += loss.item() * bs
      tot_main += loss_main.item() * bs
      tot_cnt += loss_cnt.item() * bs
      tot_f1 += f1 * bs
      n += bs

    self.sched.step()
    return {
        "loss": tot_loss / n, "loss_main": tot_main / n,
        "loss_cnt": tot_cnt / n, "f1_frame": tot_f1 / n,
        "lr": self.opt.param_groups[0]["lr"],
        "epoch": epoch,
    }
