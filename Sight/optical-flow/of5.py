import cv2
import numpy as np

# Input path
input_path = r"C:\Workspace\ProgProj\S-Sense-Project\Sight\optical-flow\data\tests-jg2\neck-IN.mp4"
cap = cv2.VideoCapture(input_path)

# Swallow counter
swallow_count = 0
swallow_active = False  # state machine: True = currently "up" (disappeared)

# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

# === Step 1: Let user draw bounding box ===
ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    cap.release()
    exit()

roi_box = cv2.selectROI("Draw ROI around Adam's apple", first_frame, False, False)
cv2.destroyWindow("Draw ROI around Adam's apple")

x, y, w, h = map(int, roi_box)

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_curr = gray[y:y+h, x:x+w]

    # Motion detection in ROI
    diff = cv2.absdiff(roi_prev, roi_curr)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Largest motion contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    adams_present = False
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:  # ignore small blobs
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x
                cy = int(M["m01"] / M["m00"]) + y
                adams_present = True
                cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

    # === State machine: count when presence disappears ===
    if not adams_present and not swallow_active:
        swallow_count += 1
        swallow_active = True
        print(f"Swallow detected! Count = {swallow_count}")

    if adams_present:
        swallow_active = False  # reset once it's back

    # Draw ROI box
    color = (0, 255, 0) if adams_present else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Counter overlay
    cv2.putText(frame, f"Swallows: {swallow_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Swallow Detector", frame)
    prev_gray = gray.copy()

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
