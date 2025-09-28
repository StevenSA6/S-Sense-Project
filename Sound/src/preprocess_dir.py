from omegaconf import OmegaConf, DictConfig
from dataio.preprocess import preprocess_directory


def main():
  cfg = OmegaConf.load("configs/initial_config.yaml")
  assert isinstance(cfg, DictConfig)
  paths = preprocess_directory(
      cfg.paths.audio_dir,
      cfg,
      "outputs/tmp",
      pattern="**/*.wav",
  )
  print(f"Processed {len(paths)} files:", paths)


if __name__ == "__main__":
  main()
