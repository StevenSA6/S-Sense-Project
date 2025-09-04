from pathlib import Path
from typing import List, Dict, Any


def scan_audio_dir(audio_dir: str) -> List[Dict[str, Any]]:
  exts = {".wav", ".flac", ".ogg", ".mp3"}
  items: List[Dict[str, Any]] = []
  for p in sorted(Path(audio_dir).glob("*")):
    if p.suffix.lower() in exts:
      # infer subject from filename prefix before first "_" else use stem
      name = p.stem
      subj = name.split("_")[0] if "_" in name else name
      items.append({"path": str(p), "subject": subj})
  return items
