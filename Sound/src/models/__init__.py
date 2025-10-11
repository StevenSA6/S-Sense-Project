from typing import Any, Dict, Union
from omegaconf import DictConfig, OmegaConf
from .sed_crnn import CRNN
from .utils import infer_in_channels
from .pann_backbones import load_pretrained_backbone


Cfg = Union[Dict[str, Any], DictConfig]


def _to_plain(cfg: Cfg) -> Dict[str, Any]:
  if isinstance(cfg, DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
  assert isinstance(cfg, dict)
  return cfg


def build_model(cfg: dict | DictConfig, return_backbone=False):
  if not isinstance(cfg, DictConfig):
    cfg = OmegaConf.create(cfg)

  backbone = None
  # Check if backbone is enabled
  backbone_cfg = cfg.get("model", {}).get("backbone", {})
  if backbone_cfg.get("enabled", False):
    backbone_type = backbone_cfg.get("type")
    if backbone_type is None:
      raise ValueError(
          "model.backbone.type must be specified when backbone is enabled")

    checkpoint_path = backbone_cfg.get("checkpoint_path", "./backbones/")
    freeze = backbone_cfg.get("freeze", True)

    # Load pretrained backbone
    backbone = load_pretrained_backbone(
        backbone_type=backbone_type,
        checkpoint_path=checkpoint_path,
        freeze=freeze
    )
    print(f"Loaded pretrained {backbone_type} backbone (frozen={freeze})")

  in_ch = infer_in_channels(cfg)
  cfg.setdefault("model", {})
  cfg["model"]["in_ch"] = in_ch

  # Pass backbone to CRNN model
  model = CRNN(cfg, backbone=backbone)

  if return_backbone:
    return model, backbone
  return model
