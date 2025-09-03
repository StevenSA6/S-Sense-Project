from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
  def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
    super().__init__()
    self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

  def forward(self, logits: torch.Tensor, targets: torch.Tensor):
    p = torch.sigmoid(logits)
    pt = p*targets + (1-p)*(1-targets)
    w = self.alpha*targets + (1-self.alpha)*(1-targets)
    loss = -w*((1-pt)**self.gamma)*torch.log(pt.clamp_min(1e-8))
    return loss.mean() if self.reduction == "mean" else loss.sum()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
  p = torch.sigmoid(logits)
  inter = (p*targets).sum(dim=1)
  denom = p.sum(dim=1) + targets.sum(dim=1) + eps
  return 1 - (2*inter + eps)/denom


def sed_loss(
    logits: torch.Tensor,         # (B,T)
    targets: torch.Tensor,        # (B,T)
    loss_cfg: Dict
):
  t = targets
  if loss_cfg.get("type", "focal") == "bce":
    pos_w = loss_cfg.get("pos_weight", 0.0)
    if pos_w and pos_w > 0:
      pw = torch.tensor([pos_w], device=logits.device)
      bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
      bce = nn.BCEWithLogitsLoss()
    main = bce(logits, t)
  else:
    main = FocalLoss(alpha=loss_cfg.get("focal_alpha", 0.25),
                     gamma=loss_cfg.get("focal_gamma", 2.0))(logits, t)
  dl_w = loss_cfg.get("dice_weight", 0.0)
  if dl_w > 0:
    main = main + dl_w * dice_loss(logits, t).mean()
  return main


def count_losses(
    out: Dict,                 # model outputs dict
    count_targets: Optional[torch.Tensor],   # (B,)
    logits: torch.Tensor,      # (B,T)
    hop: int,
    sr: int,
    cfg: Dict
):
  total = torch.tensor(0.0, device=logits.device)
  if cfg["model"]["heads"].get("count", {}).get("enabled", False) and count_targets is not None:
    if "log_lambda" in out:
      # Poisson NLL with log_lambda
      loss_cnt = F.poisson_nll_loss(torch.zeros_like(
          out["log_lambda"]), out["log_lambda"], full=False, log_input=True, reduction='mean')
      # Use target by shifting baseline: NLL(k; log_lambda) = exp(log_lambda) - k*log_lambda + log(k!)
      # Implement directly:
      lam = out["log_lambda"].exp()
      k = count_targets
      loss_cnt = (lam - k*out["log_lambda"]).mean()
    else:
      pred = out["count_mean"]
      loss_cnt = F.smooth_l1_loss(pred, count_targets)
    total = total + \
        cfg["model"]["heads"]["count"].get("loss_weight", 0.2)*loss_cnt

  if cfg["loss"].get("count_consistency", {}).get("enabled", False) and count_targets is not None:
    dt = hop / float(sr)
    count_est = torch.sigmoid(logits).sum(dim=1)*dt
    w = cfg["loss"]["count_consistency"].get("weight", 0.1)
    total = total + w * F.smooth_l1_loss(count_est, count_targets)
  return total
