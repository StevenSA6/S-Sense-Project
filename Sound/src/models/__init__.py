from omegaconf import OmegaConf
from .sed_crnn import CRNN


def infer_in_channels(cfg) -> int:
  base = 1 + (cfg.features.deltas.order if cfg.features.deltas.enabled else 0)
  aux = 0
  if cfg.features.aux_channels.enabled:
    flags = [
        cfg.features.aux_channels.spectral_flux,
        cfg.features.aux_channels.centroid,
        cfg.features.aux_channels.zcr,
        cfg.features.aux_channels.percussive_mask_mean,
        cfg.features.aux_channels.envelope_rms,
    ]
    aux = sum(bool(f) for f in flags)
  return base + aux


def build_model(cfg: dict):
  # or pass DictConfig directly if you prefer
  in_ch = infer_in_channels(OmegaConf.create(cfg))
  cfg.setdefault("model", {})
  cfg["model"]["in_ch"] = in_ch
  return CRNN(cfg)
