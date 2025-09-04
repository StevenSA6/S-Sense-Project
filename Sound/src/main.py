from omegaconf import OmegaConf, DictConfig
from train.loop import train_one_fold
from train.tester import evaluate_fold


def main():
  cfg = OmegaConf.load("configs/initial_config.yaml")
  assert isinstance(cfg, DictConfig)
  folds = cfg.cv.folds if cfg.cv.enabled else 1
  for f in range(folds):
    print(f"=== Fold {f}/{folds} ===")
    if getattr(cfg, "baseline", None) and getattr(cfg.baseline, "enabled", False):
      metrics = evaluate_fold(cfg, fold_id=f, model=None)
      print({f"fold{f}/{k}": v for k, v in metrics.items()})
    else:
      res = train_one_fold(cfg, fold_id=f)
      assert res is not None
      _, model = res
      metrics = evaluate_fold(cfg, fold_id=f, model=model)
      print({f"fold{f}/{k}": v for k, v in metrics.items()})


if __name__ == "__main__":
  main()
