import numpy as np
import cv2
import os

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

dataset_path = os.path.join("Criminal_Face_detection", "foldername", "Data_set")

face_data = []
labels = []
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.lower().endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(os.path.join(dataset_path, fx))
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)

        class_id += 1

flat_face_data = []
for data_item in face_data:
    for img in data_item:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat_face_data.append(img.flatten())

face_dataset = np.array(flat_face_data)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

print("Dataset Shape:", face_dataset.shape)
print("Labels Shape:", face_labels.shape)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    os.path.join("Criminal_Face_detection", "foldername", "haarcascade_frontalface_alt.xml")
)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        offset = 5
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)

        face_section = frame[y1:y2, x1:x2]
        face_section = cv2.resize(face_section, (100, 100))
        face_section = cv2.cvtColor(face_section, cv2.COLOR_BGR2GRAY)
        face_section = face_section.flatten()

        out = knn(trainset, face_section)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, names[int(out)], (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Criminal Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
