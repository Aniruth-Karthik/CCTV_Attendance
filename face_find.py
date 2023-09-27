from ultralytics import YOLO
import cv2
import math 
from math import atan2, degrees
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolov8n-face.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes
        
        for box in boxes:    
            cls = model.names[int(box.cls[0])]

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "face", [x1, y1], font, 1, (0,0,0), 1)
            
            cx,cy,w,h = box.xywh[0]
            face_center = (cx, cy)
            dx = w
            dy = h
            focal_length = 80
            yaw = degrees(atan2(face_center[0] - img.shape[1] / 2, focal_length))
            pitch = degrees(atan2(face_center[1] - img.shape[0] / 2, focal_length))

            eyes_center = ((x1 + w // 4 + x1 + 3 * w // 4) // 2, (y1 + h // 4 + y1 + 3 * h // 4) // 2)
            dx_roll = face_center[0] - eyes_center[0]
            dy_roll = face_center[1] - eyes_center[1]
            roll = degrees(atan2(dy_roll, dx_roll))

            cv2.putText(img, f"Yaw: {yaw:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Pitch: {pitch:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Roll: {roll:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            anomaly_threshold_pitch = 40
            if abs(pitch) > anomaly_threshold_pitch:
                cv2.putText(img, "Head Bend Down (Anomaly)", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (255, 0, 255), 3)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()