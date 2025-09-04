from pathlib import Path
from dataio.dataset import load_events_csv


def scan_audio_dir(audio_dir: str, timestamps_dir: str) -> list[dict]:
  exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
  audio_root = Path(audio_dir)
  items = []
  for p in sorted(audio_root.rglob("*")):
    if p.suffix.lower() not in exts:
      continue
    subj = p.parent.name  # parent folder = subject
    rel = p.relative_to(audio_root).with_suffix(".csv")
    csv_path = Path(timestamps_dir) / rel
    events = load_events_csv(csv_path) if csv_path.exists() else []
    items.append({"path": str(p), "subject": subj, "events": events})
  return items
