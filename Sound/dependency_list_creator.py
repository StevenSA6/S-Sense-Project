import pandas as pd
import yaml
import json

WITH_VERSION = False


def main():
  # micromamba list --json > env_packages.json
  with open("env_packages.json") as f:
    data = json.load(f)

  df = pd.DataFrame(data)
  df = df[["name", "channel", "version"]]

  env_yaml = {
      "dependencies": [
          f"{row.channel}::{row.name}{'=' + str(row.version) if WITH_VERSION else ''}"
          for row in df.itertuples(index=False)
      ]
  }

  with open("tmp.yaml", "w") as f:
    yaml.dump(env_yaml, f, sort_keys=False)


if __name__ == "__main__":
  main()
