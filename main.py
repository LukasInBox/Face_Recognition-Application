import cv2
import numpy as np
import os
from datetime import datetime
from tkinter import Tk, Button, Canvas, messagebox
from PIL import Image, ImageTk

# Ensure that folder exists to save data, it should be named "saved_attributes"
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
        
        # Button that lets the user take a snapshot and save it
        self.btn_snapshot = Button(window, text="Signup", width=50, command=self.save_snapshot)
        self.btn_snapshot.pack(anchor='center', expand=True)
        
        # Button for login
        self.btn_login = Button(window, text="Login", width=50, command=self.login)
        self.btn_login.pack(anchor='center', expand=True)

    def start(self):
        self.update()
        self.window.mainloop()
    
    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.read()
        
        if ret:
            self.display_frame(frame)
        self.window.after(10, self.update)
    
    def display_frame(self, frame):
        # Perform face detection
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
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
    
    def save_snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            face = self.detect_face(frame)
            if face is not None:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f'saved_attributes/face_{timestamp}.jpg', face)
                print("Face saved.")

    def detect_face(self, frame):
        # Extracts the face from the frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                return frame[startY:endY, startX:endX]
        return None

    def login(self):
        ret, frame = self.cap.read()
        if ret:
            login_face = self.detect_face(frame)
            if login_face is None:
                messagebox.showerror("Login failed", "No face detected.")
                return

            # Compare detected face with saved faces
            for filename in os.listdir('saved_attributes'):
                saved_face = cv2.imread(os.path.join('saved_attributes', filename))
                if self.compare_faces(saved_face, login_face):
                    messagebox.showinfo("Login Success", "Face recognized!")
                    return
            messagebox.showerror("Login failed", "Face not recognized.")

    def compare_faces(self, saved_face, login_face):
        # Simple comparison of two face images using template matching
        # Note: This is not a reliable method for actual face recognition.
        saved_face = cv2.resize(saved_face, (100, 100))
        login_face = cv2.resize(login_face, (100, 100))
        res = cv2.matchTemplate(saved_face, login_face, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Set a threshold for deciding a match
        if np.max(res) > threshold:
            return True
        return False

def main():
    root = Tk()
    app = App(root, "Tkinter and OpenCV")
    app.start()

if __name__ == '__main__':
    main()
