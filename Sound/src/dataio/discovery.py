from pathlib import Path
from typing import List, Dict, Any


def scan_audio_dir(audio_dir: str, mode: str = "prefix") -> List[Dict[str, Any]]:
  """Scan ``audio_dir`` for audio files.

  Parameters
  ----------
  audio_dir:
      Root directory containing audio files.
  mode:
      Determines how the ``subject`` field is inferred for each file.
      - ``"prefix"``: infer subject from filename prefix before the first ``_``.
      - ``"parent_dir"``: infer subject from the file's parent directory name.

  Returns
  -------
  List[Dict[str, Any]]
      List of dataset items with ``path`` and ``subject`` keys.
  """

  exts = {".wav", ".flac", ".ogg", ".mp3", "m4a"}
  items: List[Dict[str, Any]] = []
  paths = Path(audio_dir).rglob(
      "*") if mode == "parent_dir" else Path(audio_dir).glob("*")
  for p in sorted(paths):
    if p.is_file() and p.suffix.lower() in exts:
      if mode == "parent_dir":
        subj = p.parent.name
      else:  # default "prefix"
        name = p.stem
        subj = name.split("_")[0] if "_" in name else name
      items.append({"path": str(p), "subject": subj})
  return items
