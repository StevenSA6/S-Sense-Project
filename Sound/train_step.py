from typing import Dict, Optional
import torch
from sed_crnn import CRNN
from losses import sed_loss, count_losses

class Stepper:
  def __init__(self, cfg: Dict):
    self.cfg = cfg
    self.net = CRNN(cfg).to(cfg["hardware"]["device"])
    self.opt = torch.optim.AdamW(self.net.parameters(), # pyright: ignore[reportPrivateImportUsage]
                                 lr=cfg["optimizer"]["lr"],
                                 weight_decay=cfg["optimizer"]["weight_decay"])
    self.grad_clip = cfg["optimizer"].get("grad_clip_norm", 1.0)
    self.device = cfg["hardware"]["device"]

  def step(self,
           xb: torch.Tensor,          # (B,C,F,T)
           yb: torch.Tensor,          # (B,T)
           count_b: Optional[torch.Tensor],  # (B,) or None
           hop: int, sr: int) -> Dict:
    xb, yb = xb.to(self.device), yb.to(self.device)
    if count_b is not None:
      count_b = count_b.to(self.device)

    out = self.net(xb)
    logits = out["logits"]
    loss_main = sed_loss(logits, yb, self.cfg["loss"]["sed"])
    loss_cnt = count_losses(out, count_b, logits, hop, sr, self.cfg)
    loss = loss_main + loss_cnt

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
    self.opt.step()

    with torch.no_grad():
      prob = torch.sigmoid(logits)
      thr = 0.5
      f1_num = ((prob > thr) & (yb > 0.5)).sum().item()*2
      f1_den = (prob > thr).sum().item() + (yb > 0.5).sum().item() + 1e-9
      f1 = f1_num / f1_den

    return {"loss": float(loss.item()), "loss_main": float(loss_main.item()),
            "loss_cnt": float(loss_cnt.item()), "f1_frame": float(f1)}
