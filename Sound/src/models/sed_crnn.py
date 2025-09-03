from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(norm: str, ch: int):
  if norm == "group":
    g = min(32, ch)
    while ch % g != 0 and g > 1:
      g -= 1
    return nn.GroupNorm(g, ch)
  return nn.BatchNorm2d(ch)


class SE(nn.Module):
  def __init__(self, ch: int, r: int = 16):
    super().__init__()
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
        nn.Conv2d(ch, ch//r, 1), nn.SiLU(),
        nn.Conv2d(ch//r, ch, 1), nn.Sigmoid()
    )

  def forward(self, x): return x * self.fc(self.pool(x))


class DSConv(nn.Module):
  def __init__(self, c_in, c_out, k=(3, 3), s=(1, 1), norm="batch", p=0.0, se=True):
    super().__init__()
    self.dw = nn.Conv2d(c_in, c_in, k, s, padding="same",
                        groups=c_in, bias=False)
    self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
    self.bn = make_norm(norm, c_out)
    self.act = nn.SiLU()
    self.se = SE(c_out) if se else nn.Identity()
    self.do = nn.Dropout(p)

  def forward(self, x):
    x = self.dw(x)
    x = self.pw(x)
    x = self.bn(x)
    x = self.act(x)
    x = self.se(x)
    x = self.do(x)
    return x


class PosEnc(nn.Module):
  def __init__(self, d_model: int, max_len: int = 20000):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(pos*div)
    pe[:, 1::2] = torch.cos(pos*div)
    self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, H)

  def forward(self, x):  # (B,T,H)
    return x + self.pe[:, :x.size(1), :]


class CRNN(nn.Module):
  def __init__(self, cfg: Dict):
    super().__init__()
    m = cfg["model"]
    in_ch = int(m.get("in_channels", 1))
    hid = int(m.get("hidden", 256))
    p = float(m.get("dropout", 0.2))
    norm = m.get("norm", "batch")
    se_blocks = bool(m.get("se_blocks", True))

    self.cnn = nn.Sequential(
        DSConv(in_ch, 32, norm=norm, p=p, se=se_blocks),
        DSConv(32, 64, s=(2, 1), norm=norm, p=p, se=se_blocks),
        DSConv(64, 96, norm=norm, p=p, se=se_blocks),
        DSConv(96, 128, s=(2, 1), norm=norm, p=p, se=se_blocks),
    )
    self.freq_proj = nn.Conv2d(128, hid, kernel_size=(
        1, 1), bias=False)  # after collapsing F' via pooling
    self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse freq

    enc_type = m.get("rnn", {}).get("type", "lstm")
    self.enc_type = enc_type
    if enc_type == "lstm":
      rnn_h = int(m["rnn"].get("hidden", 128))
      layers = int(m["rnn"].get("layers", 2))
      bi = bool(m["rnn"].get("bidirectional", True))
      self.temporal = nn.LSTM(hid, rnn_h, num_layers=layers, batch_first=True,
                              bidirectional=bi, dropout=p if layers > 1 else 0.0)
      enc_out = rnn_h*(2 if bi else 1)
    else:
      layers = int(m["transformer"].get("layers", 2))
      nhead = int(m["transformer"].get("nhead", 4))
      dim_ff = int(m["transformer"].get("dim_ff", 4*hid))
      self.pos = PosEnc(hid)
      enc_layer = nn.TransformerEncoderLayer(
          d_model=hid, nhead=nhead, dim_feedforward=dim_ff, dropout=p, batch_first=True, activation="gelu", norm_first=True)
      self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
      enc_out = hid

    self.head_sed = nn.Linear(enc_out, 1)
    # optional duration regression per frame
    self.head_dur = nn.Linear(enc_out, 1) if m["heads"].get(
        "duration", {"enabled": False}).get("enabled", False) else None
    # optional global count head
    if m["heads"].get("count", {"enabled": False}).get("enabled", False):
      self.count_dist = m["heads"]["count"].get("dist", "poisson")
      self.head_cnt = nn.Sequential(
          nn.AdaptiveAvgPool1d(1),)  # pooling placeholder
      self.mlp_cnt = nn.Sequential(
          nn.Linear(enc_out, enc_out//2), nn.GELU(), nn.Linear(enc_out//2, 1))
    else:
      self.head_cnt, self.mlp_cnt, self.count_dist = None, None, None

    self._init_weights()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight) if isinstance(
            m, nn.Conv2d) else nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
          assert isinstance(
              m.bias, torch.Tensor), f"m.bias should be of type Tensor but got f{type(m.bias)}"
          nn.init.zeros_(m.bias)

  def forward(self, x):  # x: (B,C,F,T)
    h = self.cnn(x)                      # (B,128,F',T)
    h = self.freq_pool(h)                # (B,128,1,T)
    h = self.freq_proj(h).squeeze(2)     # (B,H,T)
    h = h.transpose(1, 2)                 # (B,T,H)
    if self.enc_type == "lstm":
      h, _ = self.temporal(h)           # (B,T,E)
    else:
      h = self.temporal(self.pos(h))   # (B,T,E)
    logits = self.head_sed(h).squeeze(-1)     # (B,T)
    out = {"logits": logits}
    if self.head_dur is not None:
      out["dur"] = F.relu(self.head_dur(h).squeeze(-1))  # (B,T)
    if self.head_cnt is not None:
      # global pooling over time
      g = h.mean(dim=1)                # (B,E)
      assert self.mlp_cnt is not None, f"self.mlp_cnt should not be None"
      lam = self.mlp_cnt(g)            # (B,1)
      if self.count_dist == "poisson":
        out["log_lambda"] = torch.log1p(
            F.softplus(lam)).squeeze(-1)  # stable >0
      else:
        out["count_mean"] = F.softplus(lam).squeeze(-1)
    return out
