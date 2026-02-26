import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("Criminal_Face_detection/foldername/haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "Criminal_Face_detection/foldername/Data_set"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

file_name = input("Enter the name of the person: ")
crime = input("Enter the crime of the person: ")
age = input("Enter the age of the person: ")

file_name = file_name + "_" + crime + "_" + age

while True:
    ret , frame = cap.read()

    if ret == False:
        continue
    
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame , 1.3 , 5)
    if len(faces) == 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    k = 1
    faces = sorted(faces, key = lambda x: x[2]*x[3] , reverse = True)

    skip += 1

    for face in faces[:1]:
        x,y,w,h = face

        offset = 5
        y1 = max(0, y-offset)
        x1 = max(0, x-offset)

        face_offset = frame[y1:y+h+offset , x1:x+w+offset]
        face_selection = cv2.resize(face_offset,(100 , 100))

        face_selection = cv2.cvtColor(face_selection, cv2.COLOR_BGR2GRAY)

        if skip % 10 == 0:
            face_data.append(face_selection)
            print(len(face_data))

        cv2.imshow("Face", face_selection)
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) , 2)

    cv2.imshow("Frame", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
face_data = np.array(face_data)

if len(face_data) != 0:
    face_data = face_data.reshape((face_data.shape[0] , -1))
    print(face_data.shape)

    save_path = os.path.join(dataset_path, file_name + ".npy")
    np.save(save_path , face_data)

    print("Dataset saved at :", save_path)
else:
    print("No face data collected")

cap.release()
cv2.destroyAllWindows()