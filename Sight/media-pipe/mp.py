import cv2
import mediapipe as mp
import numpy as np

# ---------- Config ----------
input_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\optical-flow\data\tests-jg2\neck-IN.mp4"

# Detection thresholds
HIGH_THRESH = 0.08   # fraction of face height for upward deflection
LOW_THRESH  = 0.04   # fraction for reset
REFRACTORY_SEC = 1.2 # min time between swallows

# ----------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
delay = int(1000 / fps)
refractory_frames = int(fps * REFRACTORY_SEC)

# Event state
swallow_count = 0
frames_since_last = 9999
baseline_y = None
swallow_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔄 Rotate frame if needed (change to COUNTERCLOCKWISE if wrong)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    throat_y = None
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Use chin + jawline landmarks
        idxs = [152, 148, 377, 400]
        ys = [landmarks[i].y for i in idxs]
        throat_y = np.mean(ys) * h

        # Draw dots
        for i in idxs:
            x_px, y_px = int(landmarks[i].x * w), int(landmarks[i].y * h)
            cv2.circle(frame, (x_px, y_px), 3, (255, 0, 0), -1)

    if throat_y is not None:
        if baseline_y is None:
            baseline_y = throat_y

        delta = (baseline_y - throat_y) / h  # normalized upward displacement

        # Hysteresis swallow detection
        if not swallow_active:
            if delta > HIGH_THRESH and frames_since_last > refractory_frames:
                swallow_count += 1
                swallow_active = True
                frames_since_last = 0
                print(f"Swallow detected! Count = {swallow_count}")
                # Reset baseline after swallow
                baseline_y = throat_y
        else:
            if delta < LOW_THRESH:
                swallow_active = False

        frames_since_last += 1

        # Draw baseline and current position
        cv2.line(frame, (0, int(baseline_y)), (w, int(baseline_y)), (0, 255, 255), 1)
        cv2.line(frame, (0, int(throat_y)), (w, int(throat_y)), (255, 255, 255), 1)

    # Draw swallow count
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 240, 40), 2)

    cv2.imshow("Swallow Detector (MediaPipe + Upright Video)", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
