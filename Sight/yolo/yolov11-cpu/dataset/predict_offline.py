import os
from ultralytics import YOLO

# Force offline mode to disable SSL
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["ULTRALYTICS_HUB"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Paths 
model_path = r"runs\detect\train\weights\best.pt"  # your trained weights
video_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\split-frames\tests-jg3\face-IN.mp4"

# Load model 
model = YOLO(model_path)

# Run prediction 
results = model.predict(source=video_path, conf=0.4, save=True, show=False)

print("Done! Annotated video saved in 'runs/predict/'.")
