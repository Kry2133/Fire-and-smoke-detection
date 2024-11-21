from ultralytics import YOLO
import cvzone
import cv2
import math

# Load your trained models for fire and smoke detection
fire_model_path = r"C:\Users\Admin\Downloads\Fireandsmoke combined\Fire_Detector_model\fire.pt"
smoke_model_path = r"C:\Users\Admin\Downloads\Fireandsmoke combined\SmokeDetector.v1i.yolov8.zip (Unzipped Files)-20241118T063221Z-001\SmokeDetector.v1i.yolov8.zip (Unzipped Files)\train2\weights\best.pt"

# Initialize both models
fire_model = YOLO(fire_model_path)
smoke_model = YOLO(smoke_model_path)

# Video capture
cap = cv2.VideoCapture(0)

# Class names for both models
fire_classnames = ['fire']
smoke_classnames = ['smoke']

# Confidence thresholds for detection
fire_confidence_threshold = 0.55
smoke_confidence_threshold = 0.70

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Resize the frame for consistency
    frame = cv2.resize(frame, (640, 480))

    # Run fire detection
    fire_result = fire_model(frame, stream=True)

    # Run smoke detection
    smoke_result = smoke_model(frame, stream=True)

    # Process fire detection results
    for info in fire_result:
        boxes = info.boxes
        for box in boxes:
            confidence = math.ceil(box.conf[0] * 100)
            Class = int(box.cls[0])

            # Check for valid class index
            if Class >= len(fire_classnames):
                continue

            # If confidence is above the threshold, draw a red box for fire
            if confidence > fire_confidence_threshold * 100:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Post-processing filter: Size or shape
                width, height = x2 - x1, y2 - y1
                aspect_ratio = width / height
                if width < 50 or aspect_ratio > 2.0:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red box for fire
                cvzone.putTextRect(frame, f'{fire_classnames[Class]} {confidence}%', [x1 + 8, y1 + 20],
                                   scale=1.5, thickness=2)

    # Process smoke detection results
    for info in smoke_result:
        boxes = info.boxes
        for box in boxes:
            confidence = math.ceil(box.conf[0] * 100)
            Class = int(box.cls[0])

            # Check for valid class index
            if Class >= len(smoke_classnames):
                continue

            # If confidence is above the threshold, draw a blue box for smoke
            if confidence > smoke_confidence_threshold * 100:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Post-processing filter: Size or shape
                width, height = x2 - x1, y2 - y1
                aspect_ratio = width / height
                if width < 50 or aspect_ratio > 2.0:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Blue box for smoke
                cvzone.putTextRect(frame, f'{smoke_classnames[Class]} {confidence}%', [x1 + 8, y1 + 20],
                                   scale=1.5, thickness=2)

    # Display the frame with both fire and smoke detections
    cv2.imshow('frame', frame)

    # Press 's' to stop the webcam feed
    if cv2.waitKey(1) == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
