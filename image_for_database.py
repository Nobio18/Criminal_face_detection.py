import cv2
import numpy as np
import os

name = input("Enter person name: ")
crime = input("Enter your crime: ")
age = input("Enter your age: ")

dataset_path = os.path.join("Data_set")
os.makedirs(dataset_path, exist_ok=True)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

face_data = []
count = 0
max_faces = 100 

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        offset = 5
        y1 = max(0, y-offset)
        y2 = min(frame.shape[0], y+h+offset)
        x1 = max(0, x-offset)
        x2 = min(frame.shape[1], x+w+offset)

        face_section = gray[y1:y2, x1:x2]
        face_section = cv2.resize(face_section, (100, 100))
        face_section = cv2.filter2D(face_section, -1, sharpen_kernel)

        face_data.append(face_section)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, f"{count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        count += 1

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_faces:
        break

safe_name = name.replace(" ", "_")
safe_crime = crime.replace(" ", "_")
safe_age = age.replace(" ", "_")
filename = f"{safe_name}__{safe_crime}__{safe_age}.npy"

face_data = np.array(face_data)
np.save(os.path.join(dataset_path, filename), face_data)
print(f"Saved {face_data.shape[0]} faces for {name} | Crime: {crime} | Age: {age}")

cap.release()
cv2.destroyAllWindows()