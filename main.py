import cv2
import numpy as np
import os
from datetime import datetime
from tkinter import Tk, Button, Canvas
from PIL import Image, ImageTk

# Ensure the folder exists
if not os.path.exists('saved_attributes'):
    os.makedirs('saved_attributes')

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Load the model for face detection
        self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Create a canvas that can fit the above video source size
        self.canvas = None
        
        # Update frames
        self.update()
        
        # Button that lets the user take a snapshot
        self.btn_snapshot = Button(window, text="Signup", width=50, command=self.save_snapshot)
        self.btn_snapshot.pack(anchor='center', expand=True)
        
        self.window.mainloop()
    
    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.read()
        
        if ret:
            # Perform face detection
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if self.canvas is None:
                self.canvas = Canvas(self.window, width=self.photo.width(), height=self.photo.height())
                self.canvas.pack()
            
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        
        self.window.after(10, self.update)
    
    def save_snapshot(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Perform face detection again to ensure we are saving the face correctly
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.net.setInput(blob)
                detections = self.net.forward()
                
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        face = frame[startY:endY, startX:endX]
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        cv2.imwrite(f'saved_attributes/face_{timestamp}.jpg', face)
                        print("Face saved.")
                        break  # Save the first detected face and exit loop

# Create a window and pass it to the Application object
App(Tk(), "Tkinter and OpenCV")
