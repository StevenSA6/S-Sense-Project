from typing import Any, Dict, Union
from omegaconf import DictConfig, OmegaConf
from .sed_crnn import CRNN
from .utils import infer_in_channels


Cfg = Union[Dict[str, Any], DictConfig]


def _to_plain(cfg: Cfg) -> Dict[str, Any]:
  if isinstance(cfg, DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
  assert isinstance(cfg, dict)
  return cfg


def build_model(cfg: dict | DictConfig):
  if not isinstance(cfg, DictConfig):
    cfg = OmegaConf.create(cfg)

  in_ch = infer_in_channels(cfg)
  cfg.setdefault("model", {})
  cfg["model"]["in_ch"] = in_ch
  return CRNN(cfg)
