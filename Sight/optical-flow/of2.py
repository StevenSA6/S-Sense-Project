import cv2
import numpy as np
from collections import deque
from scipy.signal import find_peaks

# Input path to your pre-recorded neck video
input_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\optical-flow\data\tests-jg2\neck-IN.mp4"
cap = cv2.VideoCapture(input_path)

# Swallow counter
swallow_count = 0

# Get FPS from the video for correct playback speed
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    cap.release()
    exit()

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Let user draw ROI (drag a box around the neck in the popup window)
roi_box = cv2.selectROI("Select ROI (draw around neck)", first_frame, False, False)
cv2.destroyWindow("Select ROI (draw around neck)")

x, y, w, h = map(int, roi_box)

# Focus only on bottom half of ROI (ignore chin)
y = y + h // 2
h = h // 2

# Set prev_gray as grayscale first frame
prev_gray = first_gray.copy()

# Sliding window for motion signal
motion_history = deque(maxlen=150)  # ~5 seconds at 30fps

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]

    # Optical flow between previous and current ROI
    flow = cv2.calcOpticalFlowFarneback(
        roi_prev, roi_curr, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Vertical component of flow (raw measurement)
    vertical_movement = np.mean(flow[..., 1])

    # Add to motion history (acts like a signal trace)
    motion_history.append(vertical_movement)

    # Convert to numpy for analysis
    signal = np.array(motion_history)

    # Peak detection (tune parameters!)
    peaks, _ = find_peaks(signal, height=1.0, distance=15)  
    # height = min strength, distance = min frames apart

    # Count swallows (based on peaks found)
    swallow_count = len(peaks)

    # ===== VISUAL CUES =====

    # ROI rectangle (green if idle, red if peak in last few frames)
    color = (0, 255, 0)
    if len(peaks) > 0 and peaks[-1] > len(signal) - 5:  # recent peak
        color = (0, 0, 255)
        cv2.circle(frame, (60, 100), 20, (0, 0, 255), -1)

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Counter overlay
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Plot signal history as mini graph
    graph_h = 100
    graph_w = 200
    graph = np.ones((graph_h, graph_w, 3), dtype=np.uint8) * 255
    if len(signal) > 1:
        sig_norm = (signal - np.min(signal)) / (np.ptp(signal) + 1e-6)
        sig_scaled = (graph_h - 1) - (sig_norm * (graph_h - 1)).astype(int)
        for i in range(1, len(sig_scaled)):
            cv2.line(graph, (i-1, sig_scaled[i-1]), (i, sig_scaled[i]), (0, 0, 255), 1)
    frame[10:10+graph_h, -graph_w-10:-10] = graph

    # Show video
    cv2.imshow("Swallow Detector", frame)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match video playback speed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
