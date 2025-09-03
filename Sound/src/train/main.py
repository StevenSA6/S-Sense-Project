from omegaconf import OmegaConf, DictConfig
from dataio.dataset import SwallowWindowDataset, DatasetConfig, read_jsonl, collate_batch
from torch.utils.data import DataLoader

from infer.inference import infer_path
from infer.eval_metrics import event_f1_sed_eval, count_mae_mape, onset_mae, subject_macro

cfg = OmegaConf.load("configs/initial_config.yaml")

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

# Inferece + evaluation
model = None  # FIXME: load weights; eval mode inside infer_path

files = ["a.wav", "b.wav"]
preds = {}
refs = {}   # {"a.wav":[{"onset":...,"offset":...}, ...], ...}
subs = {}   # {"a.wav":"S01", "b.wav":"S02"}

for f in files:
  out = infer_path(model, cfg, f, batch_size=32)
  preds[f] = (out["onsets_s"], out["offsets_s"])

f1 = event_f1_sed_eval(
    refs, preds, onset_tol=cfg.evaluation.tolerances.onset_ms/1000)
cnt = count_mae_mape(refs, preds)
omae = onset_mae(
    refs, preds, match_tol=cfg.evaluation.tolerances.onset_ms/1000)
macro = subject_macro(
    refs, preds, subs, onset_tol=cfg.evaluation.tolerances.onset_ms/1000)

print(f1, cnt, {"OnsetMAE_s": omae}, macro)
