import cv2
import numpy as np
import os
from datetime import datetime
from tkinter import Tk, Button, Canvas, Entry, Label, messagebox
from PIL import Image, ImageTk

# Ensure that folder exists to save data, it should be named "saved_attributes"
if not os.path.exists('saved_attributes'):
    os.makedirs('saved_attributes')

# Load user statuses at startup
user_statuses = {}

def load_user_statuses():
    status_file = 'user_status.txt'
    if os.path.exists(status_file):
        with open(status_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 2:
                    username, action = parts[0], parts[-1]
                    user_statuses[username] = action

def save_user_status(username, action):
    user_statuses[username] = action
    with open('user_status.txt', 'w') as file:
        for user, act in user_statuses.items():
            file.write(f"{user} clocked {act} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_time(username, action):
    filename = f"clock_{action}_times.txt"
    with open(filename, "a") as file:
        file.write(f"{username} clocked {action} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        load_user_statuses()
        
        # Load the model for face detection
        self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Create a canvas that can fit the above video source size
        self.canvas = None

        # Entry and label for user name during signup or recognition
        self.label_user_name = Label(window, text="Enter/Recognized Username:")
        self.label_user_name.pack(anchor='center')
        self.entry_user_name = Entry(window)
        self.entry_user_name.pack(anchor='center')

        # Button that lets the user take a snapshot and save it
        self.btn_snapshot = Button(window, text="Signup", width=50, command=self.save_snapshot)
        self.btn_snapshot.pack(anchor='center', expand=True)
        
        # Button for clocking in
        self.btn_clock_in = Button(window, text="Clock In", width=50, command=lambda: self.handle_clocking("in"))
        self.btn_clock_in.pack(anchor='center', expand=True)
        
        # Button for clocking out
        self.btn_clock_out = Button(window, text="Clock Out", width=50, command=lambda: self.handle_clocking("out"))
        self.btn_clock_out.pack(anchor='center', expand=True)

        self.start()

    def start(self):
        self.update()
        self.window.mainloop()
    
    def update(self):
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

                face = frame[startY:endY, startX:endX]
                username = "Unknown"
                for filename in os.listdir('saved_attributes'):
                    saved_face = cv2.imread(os.path.join('saved_attributes', filename))
                    if self.compare_faces(saved_face, face):
                        username = filename.split('_')[0]
                        break

                text = f"{username}: {confidence * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if self.canvas is None:
            self.canvas = Canvas(self.window, width=self.photo.width(), height=self.photo.height())
            self.canvas.pack()
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def save_snapshot(self):
        username = self.entry_user_name.get().strip()
        if not username:
            messagebox.showerror("Signup Failed", "Please enter a user name.")
            return

        ret, frame = self.cap.read()
        if ret:
            face = self.detect_face(frame)
            if face is not None:
                # Check for existing faces
                face_already_registered = False
                for filename in os.listdir('saved_attributes'):
                    saved_face = cv2.imread(os.path.join('saved_attributes', filename))
                    if saved_face is not None and self.compare_faces(saved_face, face):
                        face_already_registered = True
                        break
                
                if face_already_registered:
                    messagebox.showerror("Signup Failed", "This face is already registered.")
                    return

                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f'saved_attributes/{username}_{timestamp}.jpg'
                cv2.imwrite(filename, face)
                messagebox.showinfo("Snapshot Saved", f"Face saved for user: {username}")
            else:
                messagebox.showerror("Signup Failed", "No face detected. Try again.")

    def detect_face(self, frame):
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

    def handle_clocking(self, action):
        ret, frame = self.cap.read()
        if ret:
            face = self.detect_face(frame)
            if face is None:
                messagebox.showerror("Clocking failed", "No face detected. Please ensure your face is in the frame.")
                return

            face_recognized = False
            matched_user = ""
            for filename in os.listdir('saved_attributes'):
                saved_face = cv2.imread(os.path.join('saved_attributes', filename))
                if self.compare_faces(saved_face, face):
                    face_recognized = True
                    matched_user = filename.split('_')[0]
                    self.entry_user_name.delete(0, 'end')
                    self.entry_user_name.insert(0, matched_user)
                    break

            if not face_recognized:
                messagebox.showerror("Clocking failed", "Face not recognized. You are not authorized to clock.")
                return

            # Check if user can perform the action
            last_action = user_statuses.get(matched_user, None)
            if last_action == action:
                messagebox.showerror("Clocking Error", f"User '{matched_user}' cannot clock {action} again without clocking {'out' if action == 'in' else 'in'}.")
                return

            save_user_status(matched_user, action)
            log_time(matched_user, action)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            messagebox.showinfo("Clocking Success", f"User '{matched_user}' clocked {action} successfully at {timestamp}.")

    def compare_faces(self, saved_face, login_face):
        saved_face = cv2.resize(saved_face, (100, 100))
        login_face = cv2.resize(login_face, (100, 100))
        res = cv2.matchTemplate(saved_face, login_face, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Threshold for deciding a match
        if np.max(res) > threshold:
            return True
        return False

def main():
    root = Tk()
    app = App(root, "Tkinter and OpenCV")
    app.start()

if __name__ == '__main__':
    main()
