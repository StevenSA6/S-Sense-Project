import os

# 🔒 Force offline *before* importing Ultralytics
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["ULTRALYTICS_HUB"] = "0"
os.environ["NO_COLOR"] = "1"

# ✅ Also neutralize SSL lookups globally
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

from ultralytics import YOLO

# === Paths ===
model_path = r"runs\train\exp\weights\best.pt"
data_path = r"dataset\data.yaml"

# === Run validation ===
model = YOLO(model_path)
results = model.val(data=data_path, imgsz=640)
print("✅ Offline validation complete.")
print(results)
