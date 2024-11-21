from ultralytics import YOLO
import cvzone
import cv2
import math

# Load the trained YOLO model
model_path = r"C:\Users\Admin\Downloads\SmokeDetector.v1i.yolov8.zip (Unzipped Files)-20241118T063221Z-001\SmokeDetector.v1i.yolov8.zip (Unzipped Files)\train2\weights\best.pt"  # Update with your model path
model = YOLO(model_path)

# Set up video capture (webcam or video file)
cap = cv2.VideoCapture(0)  # Change to file path for video: r"C:\path_to_your_video.mp4"

# Class name
classnames = ['smoke']

# Threshold for detection
confidence_threshold = 0.55

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Perform detection
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = math.ceil(box.conf[0] * 100)
            Class = int(box.cls[0])

            if confidence > confidence_threshold * 100:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 25],
                                   scale=1, thickness=1)

    # Show the frame
    cv2.imshow('Smoke Detection', frame)

    # Exit loop when 's' key is pressed
    if cv2.waitKey(1) == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
