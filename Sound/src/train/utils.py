from typing import Any
from sklearn.model_selection import GroupKFold, KFold
import pandas as pd
import numpy as np


def make_folds(
    items: list[dict[str, Any]],
    folds: int,
    split_by: str = "subject",
) -> list[tuple[np.ndarray, np.ndarray]]:
  # Convert items to a DataFrame for sklearn compatibility
  df = pd.DataFrame(items)

  if split_by == "subject":
    if "subject" not in df:
      raise ValueError("`subject` column not found in items")
    groups = df["subject"]
    return list(GroupKFold(n_splits=folds).split(df, groups=groups))
  else:
    return list(KFold(n_splits=folds, shuffle=True, random_state=42).split(df))
