import cv2
import numpy as np
import os
from datetime import datetime

# Ensure saves_attributes folder exists
if not os.path.exists('saved_attributes'):
    os.makedirs('saved_attributes')

# Load the model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence < 0.5:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Check if 's' key is pressed to "Signup"
    if key == ord("s"):
        if 'startX' in locals():  # Check if a face was detected
            face_img = frame[startY:endY, startX:endX]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f'saved_attributes/face_{timestamp}.jpg', face_img)
            print("Face saved.")
    
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
