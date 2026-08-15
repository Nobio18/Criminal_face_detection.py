# import cv2
# import numpy as np
# import os

# name = input("Enter person name: ")
# crime = input("Enter crime: ")
# age = input("Enter age: ")

# dataset_path = "Data_set"
# os.makedirs(dataset_path, exist_ok=True)

# person_folder = os.path.join(dataset_path, name.replace(" ", "_"))
# os.makedirs(person_folder, exist_ok=True)

# cap = cv2.VideoCapture(0)

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
# )

# sharpen_kernel = np.array([
#     [0,-1,0],
#     [-1,5,-1],
#     [0,-1,0]
# ])

# face_data = []
# count = 0
# max_faces = 50

# def is_blurry(image):
#     return cv2.Laplacian(image, cv2.CV_64F).var() < 50

# while True:

#     ret, frame = cap.read()
#     if not ret:
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x,y,w,h) in faces:

#         offset = 10
#         face_section = gray[y-offset:y+h+offset, x-offset:x+w+offset]

#         face_section = cv2.resize(face_section,(100,100))
#         face_section = cv2.filter2D(face_section,-1,sharpen_kernel)

#         if is_blurry(face_section):
#             continue

#         face_data.append(face_section)

#         file_name = os.path.join(person_folder,f"{name}_{count}.jpg")
#         cv2.imwrite(file_name, face_section)

#         count += 1

#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.putText(frame,str(count),(x,y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

#     cv2.imshow("Face Capture",frame)

#     if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_faces:
#         break

# safe_name = name.replace(" ","_")
# safe_crime = crime.replace(" ","_")
# safe_age = age.replace(" ","_")

# filename = f"{safe_name}__{safe_crime}__{safe_age}.npy"

# face_data = np.array(face_data)

# np.save(os.path.join(dataset_path,filename),face_data)

# # print(f"\nSaved {face_data.shape[0]} images for {name}")
# # print("Dataset ready for training")

# cap.release()
# cv2.destroyAllWindows()

# # #---------------------------------------------------( WITH GUI )-------------------------------------------------

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

DATASET_PATH = r"C:\My_Projects\Criminal_detection\Data_set"

BG = "#0D0D0D"
RED = "#FF0000"
GREEN = "#00FF00"
BLUE = "#00BFFF"
PANEL = "#2E2E2E"
DARK_PANEL = "#181818"
WHITE = "#FFFFFF"
TEXT = "#E8E8E8"
MUTED = "#A0A0A0"
BORDER = "#3D3D3D"
CAMERA_BG = "#141414"
BUTTON_GREY = "#707070"

root = tk.Tk()
root.title("VisionShield | Face Dataset Collection")
root.geometry("1250x720")
root.resizable(False, False)
root.configure(bg=BG)

camera = None
camera_running = False
face_data = []
count = 0
max_faces = 50

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_alt.xml"
)

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


def is_blurry(image):
    return cv2.Laplacian(
        image,
        cv2.CV_64F
    ).var() < 50


def start_camera():
    global camera
    global camera_running
    global face_data
    global count

    name = name_entry.get().strip()
    crime = crime_entry.get().strip()
    age = age_entry.get().strip()

    if not name or not crime or not age:
        messagebox.showwarning(
            "Missing Information",
            "Please enter Name, Crime and Age."
        )
        return

    if not age.isdigit():
        messagebox.showwarning(
            "Invalid Age",
            "Please enter a valid age."
        )
        return

    os.makedirs(
        DATASET_PATH,
        exist_ok=True
    )

    face_data = []
    count = 0

    count_label.config(
        text="Images Captured: 0/50"
    )

    progress_label.config(
        text="Progress: 0%"
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        messagebox.showerror(
            "Camera Error",
            "Unable to access camera."
        )
        camera = None
        return

    camera_running = True

    camera_status.config(
        text="● CAMERA ACTIVE",
        fg=GREEN
    )

    status_label.config(
        text="CAPTURING IMAGES...",
        fg=BLUE
    )

    start_button.config(
        state="disabled",
        bg="#174C61"
    )

    stop_button.config(
        state="normal",
        bg=BUTTON_GREY,
        fg=WHITE
    )

    save_button.config(
        state="disabled",
        bg="#505050",
        fg="#BDBDBD"
    )

    update_camera()


def update_camera():
    global count

    if not camera_running or camera is None:
        return

    ret, frame = camera.read()

    if not ret:
        root.after(
            15,
            update_camera
        )
        return

    frame = cv2.flip(
        frame,
        1
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(70, 70)
    )

    for x, y, w, h in faces:

        if count >= max_faces:
            break

        offset = 10

        y1 = max(
            0,
            y - offset
        )

        y2 = min(
            gray.shape[0],
            y + h + offset
        )

        x1 = max(
            0,
            x - offset
        )

        x2 = min(
            gray.shape[1],
            x + w + offset
        )

        face_section = gray[
            y1:y2,
            x1:x2
        ]

        if face_section.size == 0:
            continue

        face_section = cv2.resize(
            face_section,
            (100, 100)
        )

        face_section = cv2.filter2D(
            face_section,
            -1,
            sharpen_kernel
        )

        if is_blurry(face_section):
            continue

        face_data.append(
            face_section
        )

        count += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{count}/50",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    count_label.config(
        text=f"Images Captured: {count}/50"
    )

    progress_label.config(
        text=f"Progress: {int((count / 50) * 100)}%"
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        frame
    )

    image.thumbnail(
        (820, 450)
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    camera_view.config(
        image=photo,
        text=""
    )

    camera_view.image = photo

    if count >= max_faces:

        stop_camera()

        status_label.config(
            text="50 IMAGES CAPTURED",
            fg=GREEN
        )

        save_button.config(
            state="normal",
            bg=GREEN,
            fg=BG
        )

        info_label.config(
            text="Capture complete. Click SAVE DATASET."
        )

        return

    root.after(
        15,
        update_camera
    )


def stop_camera():
    global camera
    global camera_running

    camera_running = False

    if camera is not None:
        camera.release()
        camera = None

    camera_status.config(
        text="● CAMERA INACTIVE",
        fg=RED
    )

    start_button.config(
        state="normal",
        bg=BLUE
    )

    stop_button.config(
        state="disabled",
        bg="#505050",
        fg="#BDBDBD"
    )

    camera_view.config(
        image="",
        text="CAMERA INACTIVE\n\nSTART TO BEGIN",
        fg=MUTED,
        bg=CAMERA_BG
    )

    camera_view.image = None


def save_dataset():

    if len(face_data) == 0:
        messagebox.showwarning(
            "No Data",
            "Please capture images first."
        )
        return

    name = name_entry.get().strip()
    crime = crime_entry.get().strip()
    age = age_entry.get().strip()

    safe_name = name.replace(
        " ",
        "_"
    )

    safe_crime = crime.replace(
        " ",
        "_"
    )

    safe_age = age.replace(
        " ",
        "_"
    )

    person_folder = os.path.join(
        DATASET_PATH,
        safe_name
    )

    os.makedirs(
        person_folder,
        exist_ok=True
    )

    for index, image in enumerate(face_data):

        file_name = os.path.join(
            person_folder,
            f"{safe_name}_{index}.jpg"
        )

        cv2.imwrite(
            file_name,
            image
        )

    filename = (
        f"{safe_name}__"
        f"{safe_crime}__"
        f"{safe_age}.npy"
    )

    np.save(
        os.path.join(
            DATASET_PATH,
            filename
        ),
        np.array(face_data)
    )

    messagebox.showinfo(
        "Dataset Saved",
        f"Dataset saved successfully!\n\n"
        f"Name: {name}\n"
        f"Crime: {crime}\n"
        f"Age: {age}\n"
        f"Images: {len(face_data)}"
    )

    status_label.config(
        text="DATASET SAVED SUCCESSFULLY",
        fg=GREEN
    )

    info_label.config(
        text="Dataset is ready for training."
    )

    save_button.config(
        state="disabled",
        bg="#505050",
        fg="#BDBDBD"
    )


def reset():
    global face_data
    global count

    stop_camera()

    face_data = []
    count = 0

    name_entry.delete(
        0,
        tk.END
    )

    crime_entry.delete(
        0,
        tk.END
    )

    age_entry.delete(
        0,
        tk.END
    )

    count_label.config(
        text="Images Captured: 0/50"
    )

    progress_label.config(
        text="Progress: 0%"
    )

    status_label.config(
        text="SYSTEM READY",
        fg=GREEN
    )

    info_label.config(
        text="Enter person details and start camera."
    )

    save_button.config(
        state="disabled",
        bg="#505050",
        fg="#BDBDBD"
    )


def exit_system():
    global camera

    if camera is not None:
        camera.release()

    cv2.destroyAllWindows()
    root.destroy()


header = tk.Frame(
    root,
    bg=PANEL,
    height=60
)

header.pack(
    fill="x"
)

header.pack_propagate(False)

tk.Label(
    header,
    text="VisionShield",
    font=("Segoe UI", 19, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    side="left",
    padx=25
)

tk.Label(
    header,
    text="AI-DRIVEN FACE DATASET COLLECTION",
    font=("Segoe UI", 8, "bold"),
    fg=BLUE,
    bg=PANEL
).pack(
    side="left"
)

tk.Label(
    header,
    text="● DATA COLLECTION",
    font=("Segoe UI", 8, "bold"),
    fg=GREEN,
    bg=PANEL
).pack(
    side="right",
    padx=25
)


main = tk.Frame(
    root,
    bg=BG
)

main.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=18
)


watermark = tk.Label(
    root,
    text="DEVIL",
    font=("Segoe UI", 13),
    fg="#8A8A8A",
    bg=BG
)

watermark.place(
    relx=0.975,
    rely=0.96,
    anchor="se"
)


left_panel = tk.Frame(
    main,
    bg=PANEL,
    width=290,
    highlightbackground=BORDER,
    highlightthickness=1
)

left_panel.pack(
    side="left",
    fill="y",
    padx=(0, 16)
)

left_panel.pack_propagate(False)


tk.Label(
    left_panel,
    text="PERSON INFORMATION",
    font=("Segoe UI", 13, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    pady=(25, 20)
)


tk.Label(
    left_panel,
    text="NAME",
    font=("Segoe UI", 9, "bold"),
    fg=BLUE,
    bg=PANEL,
    anchor="w"
).pack(
    padx=25,
    fill="x"
)

name_entry = tk.Entry(
    left_panel,
    font=("Segoe UI", 11),
    bg=DARK_PANEL,
    fg=WHITE,
    insertbackground=BLUE,
    relief="flat"
)

name_entry.pack(
    padx=25,
    pady=(7, 17),
    ipady=8,
    fill="x"
)


tk.Label(
    left_panel,
    text="CRIME",
    font=("Segoe UI", 9, "bold"),
    fg=BLUE,
    bg=PANEL,
    anchor="w"
).pack(
    padx=25,
    fill="x"
)

crime_entry = tk.Entry(
    left_panel,
    font=("Segoe UI", 11),
    bg=DARK_PANEL,
    fg=WHITE,
    insertbackground=BLUE,
    relief="flat"
)

crime_entry.pack(
    padx=25,
    pady=(7, 17),
    ipady=8,
    fill="x"
)


tk.Label(
    left_panel,
    text="AGE",
    font=("Segoe UI", 9, "bold"),
    fg=BLUE,
    bg=PANEL,
    anchor="w"
).pack(
    padx=25,
    fill="x"
)

age_entry = tk.Entry(
    left_panel,
    font=("Segoe UI", 11),
    bg=DARK_PANEL,
    fg=WHITE,
    insertbackground=BLUE,
    relief="flat"
)

age_entry.pack(
    padx=25,
    pady=(7, 20),
    ipady=8,
    fill="x"
)


camera_status = tk.Label(
    left_panel,
    text="● CAMERA INACTIVE",
    font=("Segoe UI", 8, "bold"),
    fg=RED,
    bg=PANEL
)

camera_status.pack(
    pady=(0, 13)
)


start_button = tk.Button(
    left_panel,
    text="📷  START CAMERA",
    command=start_camera,
    font=("Segoe UI", 9, "bold"),
    bg=BLUE,
    fg=BG,
    activebackground="#33CCFF",
    relief="flat",
    cursor="hand2"
)

start_button.pack(
    padx=25,
    fill="x",
    ipady=7
)


stop_button = tk.Button(
    left_panel,
    text="■  STOP CAMERA",
    command=stop_camera,
    font=("Segoe UI", 9, "bold"),
    bg="#505050",
    fg="#BDBDBD",
    activebackground=BUTTON_GREY,
    relief="flat",
    cursor="hand2",
    state="disabled"
)

stop_button.pack(
    padx=25,
    pady=8,
    fill="x",
    ipady=7
)


save_button = tk.Button(
    left_panel,
    text="SAVE DATASET",
    command=save_dataset,
    font=("Segoe UI", 9, "bold"),
    bg="#505050",
    fg="#BDBDBD",
    activebackground="#33FF33",
    relief="flat",
    cursor="hand2",
    state="disabled"
)

save_button.pack(
    padx=25,
    fill="x",
    ipady=7
)


tk.Button(
    left_panel,
    text="RESET",
    command=reset,
    font=("Segoe UI", 9, "bold"),
    bg=BUTTON_GREY,
    fg=WHITE,
    activebackground="#858585",
    relief="flat",
    cursor="hand2"
).pack(
    padx=25,
    pady=8,
    fill="x",
    ipady=7
)


tk.Button(
    left_panel,
    text="EXIT SYSTEM",
    command=exit_system,
    font=("Segoe UI", 9, "bold"),
    bg=RED,
    fg=WHITE,
    activebackground="#FF3333",
    relief="flat",
    cursor="hand2"
).pack(
    padx=25,
    fill="x",
    ipady=7
)


count_label = tk.Label(
    left_panel,
    text="Images Captured: 0/50",
    font=("Segoe UI", 9, "bold"),
    fg=TEXT,
    bg=PANEL
)

count_label.pack(
    pady=(18, 4)
)


progress_label = tk.Label(
    left_panel,
    text="Progress: 0%",
    font=("Segoe UI", 8),
    fg=BLUE,
    bg=PANEL
)

progress_label.pack()


right_panel = tk.Frame(
    main,
    bg=BG
)

right_panel.pack(
    side="right",
    fill="both",
    expand=True
)


camera_card = tk.Frame(
    right_panel,
    bg=PANEL,
    highlightbackground=BLUE,
    highlightthickness=2
)

camera_card.pack(
    fill="both",
    expand=True
)


tk.Label(
    camera_card,
    text="LIVE FACE CAPTURE",
    font=("Segoe UI", 13, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    pady=13
)


camera_view = tk.Label(
    camera_card,
    text="CAMERA INACTIVE\n\nSTART TO BEGIN",
    font=("Segoe UI", 14, "bold"),
    fg=MUTED,
    bg=CAMERA_BG,
    justify="center"
)

camera_view.pack(
    padx=15,
    pady=(0, 15),
    fill="both",
    expand=True
)


status_label = tk.Label(
    right_panel,
    text="SYSTEM READY",
    font=("Segoe UI", 10, "bold"),
    fg=GREEN,
    bg=BG
)

status_label.pack(
    pady=(10, 4)
)


info_label = tk.Label(
    right_panel,
    text="Enter person details and start camera.",
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=BG
)

info_label.pack()


root.protocol(
    "WM_DELETE_WINDOW",
    exit_system
)

root.mainloop()