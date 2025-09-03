from omegaconf import OmegaConf, DictConfig
from dataset import SwallowWindowDataset, DatasetConfig, read_jsonl, collate_batch
from torch.utils.data import DataLoader

cfg = OmegaConf.load("configs/config.yaml")

train_items = read_jsonl(cfg.paths.train_manifest)
val_items = read_jsonl(cfg.paths.val_manifest)
assert isinstance(
    cfg, DictConfig), f"cfg should be of type DictConfig but got type {type(DictConfig)}"
train_ds = SwallowWindowDataset(
    train_items, DatasetConfig(cfg=cfg, train=True))
val_ds = SwallowWindowDataset(val_items,   DatasetConfig(cfg=cfg, train=False))

train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                          shuffle=True, num_workers=cfg.hardware.num_workers,
                          pin_memory=True, collate_fn=collate_batch)
val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                        shuffle=False, num_workers=cfg.hardware.num_workers,
                        pin_memory=True, collate_fn=collate_batch)
