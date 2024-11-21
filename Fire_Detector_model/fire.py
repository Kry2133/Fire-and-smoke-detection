from ultralytics import YOLO
import cvzone
import cv2
import math

model_path = r"C:\Users\Prakhar\Desktop\Fire_Detector_model\fire.pt"
#already trained this in google callab 

model = YOLO(model_path)

#cap = cv2.VideoCapture(r"C:\Users\prakh\Downloads\Fire Detector Course\fire2.mp4")
cap = cv2.VideoCapture(0) 


classnames = ['fire']

confidence_threshold = 0.55

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    
    frame = cv2.resize(frame, (640, 480))

    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = math.ceil(box.conf[0] * 100)
            Class = int(box.cls[0])

            if confidence > 60:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)


    cv2.imshow('frame', frame)
    

    if cv2.waitKey(1) == ord('s'):
        break


cap.release()
cv2.destroyAllWindows()

#To add:
#sound alarm 
#text/mail notification


