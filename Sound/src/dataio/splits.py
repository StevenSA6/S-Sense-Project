from sklearn.model_selection import GroupKFold, KFold


def make_folds(items, folds, split_by="subject"):
  if split_by == "subject":
    groups = [it["subject"] for it in items]
    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(items, groups=groups))
  else:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    return list(splitter.split(items))
