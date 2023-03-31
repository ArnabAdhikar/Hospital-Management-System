import cv2
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image
import os
#import hospital as hos
def generate_dataset():
    face_classifier = cv2.CascadeClassifier("1.xml")
    def face_cropped(img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(grey, 1.3, 5)
        # scalling factor = 1.3
        # minimum neighbour = 5
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_faces = img[y:y+h, x:x+w]
        return cropped_faces
    cam = cv2.VideoCapture(0)
    id = 1
    image_id = 0
    while True:
        ret, frame = cam.read()
        if face_cropped(frame) is not None:
            image_id = image_id + 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file = "data/user." + str(id) + "." + str(image_id) + ".jpg"
            cv2.imwrite(file, face)
            cv2.putText(face, str(image_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Faces", face)
        if cv2.waitKey(1) == 13 or int(image_id) == 200:
            break
    cam.release()
    cv2.destroyAllWindows()
    print("Collection completed.....")
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    # Train and save classifier
    clf = cv2.face_LBPHFaceRecognizer.create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
text = ""
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))
        global text
        if confidence > 55:
            if id == 1:
                text = "Anon Anderston"
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                print(text)
        else:
            text = "Unknown"
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return text, coords
def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), clf)
    return img
# loading classifier
def ultimate():
    faceCascade = cv2.CascadeClassifier("1.xml")
    clf = cv2.face_LBPHFaceRecognizer.create()
    clf.read("classifier.xml")
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face Detection", img)
        if cv2.waitKey(1) == 13:
            break
    video_capture.release()
    cv2.destroyAllWindows()

#ultimate()
#train_classifier("data")
#generate_dataset()
'''def conf():
        root = tk.Tk()
        b = hos.Hospital
        ob = b(root)
        root.mainloop()'''

def build():
    def exit():
        exit = messagebox.askyesno("Hospital Management System", "Confirm you want to exit")
        if exit > 0:
            window.destroy()
            return
    window = tk.Tk()
    window.title("Face Authentication system")
    window.geometry("800x100")
    l1 = tk.Label(window, text="Own By: Anon Anderson", font=("Algerian", 20))
    l1.grid(column=0, row=0)
    b2 = tk.Button(window, text="Detect the face", font=("Algerian", 18), bg="green", fg="orange", command=ultimate)
    b2.grid(column=0, row=2)
    b3 = tk.Button(window, text="Exit", font=("Algerian", 18), bg="green", fg="yellow", command=exit)
    b3.grid(column=3, row=2)
    window.mainloop()
    return text
