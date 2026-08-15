# # import cv2
# # import os
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.neighbors import KNeighborsClassifier
# # import tkinter as tk
# # from tkinter import messagebox

# # login_success = False

# # def login():

# #     global login_success

# #     username = username_entry.get()
# #     password = password_entry.get()

# #     if username == "Devil" and password == "1234":
# #         login_success = True
# #         root.destroy()

# #     else:
# #         messagebox.showerror(
# #             "Login Failed",
# #             "Invalid Username or Password"
# #         )


# # root = tk.Tk()

# # root.title("Criminal Detection Login")
# # root.geometry("350x220")
# # root.resizable(False, False)

# # tk.Label(
# #     root,
# #     text="Criminal Detection System",
# #     font=("Arial", 14, "bold")
# # ).pack(pady=15)

# # tk.Label(root, text="Username").pack()

# # username_entry = tk.Entry(root, width=30)
# # username_entry.pack()

# # tk.Label(root, text="Password").pack()

# # password_entry = tk.Entry(root, show="*", width=30)
# # password_entry.pack()

# # tk.Button(
# #     root,
# #     text="Login",
# #     command=login,
# #     width=15,
# #     bg="green",
# #     fg="white"
# # ).pack(pady=15)

# # root.mainloop()

# # if not login_success:
# #     exit()

# # import cv2
# # import os
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.neighbors import KNeighborsClassifier

# # dataset_path = r"C:\My_Projects\Criminal_detection\Data_set"

# # face_data = []
# # labels = []
# # names = {}

# # print("Loading Dataset...")

# # class_id = 0

# # for file in os.listdir(dataset_path):

# #     if not file.endswith(".npy"):
# #         continue

# #     parts = file[:-4].split("__")

# #     if len(parts) < 2:
# #         continue

# #     name = parts[0]
# #     crime = parts[1]

# #     names[class_id] = f"{name} | {crime}"

# #     data = np.load(os.path.join(dataset_path, file))

# #     for img in data:

# #         if len(img.shape) == 3:
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #         img = cv2.resize(img, (100, 100))
# #         face_data.append(img.flatten())
# #         labels.append(class_id)

# #     class_id += 1

# # face_data = np.array(face_data)
# # labels = np.array(labels)

# # print("Dataset Loaded")

# # X_train, X_test, y_train, y_test = train_test_split(
# #     face_data,
# #     labels,
# #     test_size=0.2,
# #     random_state=42
# # )

# # model = KNeighborsClassifier(n_neighbors=5)
# # model.fit(X_train, y_train)

# # # accuracy = model.score(X_test, y_test) * 100
# # # print(f"Accuracy: {accuracy:.2f}%")

# # face_cascade = cv2.CascadeClassifier(
# #     cv2.data.haarcascades +
# #     "haarcascade_frontalface_alt.xml"
# # )

# # cap = cv2.VideoCapture(0)

# # while True:

# #     ret, frame = cap.read()

# #     if not ret:
# #         continue

# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #     faces = face_cascade.detectMultiScale(
# #         gray,
# #         scaleFactor=1.3,
# #         minNeighbors=5
# #     )

# #     for (x, y, w, h) in faces:

# #         face = gray[y:y+h, x:x+w]

# #         try:
# #             face = cv2.resize(face, (100, 100))
# #         except:
# #             continue

# #         face = face.flatten().reshape(1, -1)

# #         prediction = model.predict(face)[0]

# #         label = names[prediction]

# #         cv2.rectangle(
# #             frame,
# #             (x, y),
# #             (x+w, y+h),
# #             (0, 255, 0),
# #             2
# #         )

# #         cv2.putText(
# #             frame,
# #             label,
# #             (x, y-10),
# #             cv2.FONT_HERSHEY_SIMPLEX,
# #             0.7,
# #             (0, 255, 0),
# #             2
# #         )

# #     cv2.imshow("Criminal Detection", frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# # #---------------------------------------------------( WITH GUI )-------------------------------------------------

import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
BUTTON_GREY_HOVER = "#858585"

root = tk.Tk()
root.title("VisionShield | AI-Driven Criminal Detection System")
root.geometry("1250x720")
root.resizable(False, False)
root.configure(bg=BG)

model = None
names = {}
camera = None
camera_running = False
last_person = None
last_crime = None


def load_model():
    global model, names

    face_data = []
    labels = []
    names = {}
    class_id = 0

    if not os.path.exists(DATASET_PATH):
        messagebox.showerror(
            "Dataset Error",
            "Dataset folder not found:\n\n" + DATASET_PATH
        )
        return False

    for file in os.listdir(DATASET_PATH):
        if not file.endswith(".npy"):
            continue

        parts = file[:-4].split("__")

        if len(parts) < 2:
            continue

        name = parts[0]
        crime = parts[1]

        names[class_id] = {
            "name": name,
            "crime": crime
        }

        data = np.load(
            os.path.join(DATASET_PATH, file)
        )

        for img in data:
            if len(img.shape) == 3:
                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

            img = cv2.resize(
                img,
                (100, 100)
            )

            face_data.append(
                img.flatten()
            )

            labels.append(class_id)

        class_id += 1

    if len(face_data) == 0:
        messagebox.showerror(
            "Dataset Error",
            "No .npy dataset files found."
        )
        return False

    if len(np.unique(labels)) < 2:
        messagebox.showerror(
            "Dataset Error",
            "At least two persons are required."
        )
        return False

    face_data = np.array(face_data)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        face_data,
        labels,
        test_size=0.2,
        random_state=42
    )

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(
        X_train,
        y_train
    )

    return True


def login():
    username = username_entry.get().strip()
    password = password_entry.get()

    if username == "Sudhanshu" and password == "1234":
        if not load_model():
            return

        login_page.pack_forget()

        dashboard.pack(
            fill="both",
            expand=True
        )

        user_label.config(
            text="User: Sudhanshu  |  VisionShield"
        )

        system_status.config(
            text="● SYSTEM ONLINE",
            fg=GREEN
        )
    else:
        messagebox.showerror(
            "Login Failed",
            "Invalid Username or Password."
        )


def start_camera():
    global camera, camera_running

    if camera_running:
        return

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
        text="SYSTEM READY – DETECTING",
        fg=BLUE
    )

    activity_label.config(
        text="Scanning for faces..."
    )

    timeline_label.config(
        text="Timeline: Idle → Detecting"
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

    update_camera()


def stop_camera():
    global camera, camera_running

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

    if last_person is None:
        status_label.config(
            text="SYSTEM READY – AWAITING INPUT",
            fg=GREEN
        )

        activity_label.config(
            text="No Activity Detected"
        )

        result_title.config(
            text="NO ACTIVITY DETECTED",
            fg=MUTED
        )

        result_person.config(
            text="Waiting for detection...",
            fg=TEXT
        )

        result_crime.config(
            text="Crime information will appear here",
            fg=MUTED
        )

        timeline_label.config(
            text="Timeline: Idle"
        )
    else:
        status_label.config(
            text="DETECTION REPORT READY",
            fg=RED
        )

        activity_label.config(
            text="Person identified successfully"
        )

        result_title.config(
            text="FINAL DETECTION REPORT",
            fg=RED
        )

        result_person.config(
            text="Person: " + last_person,
            fg=GREEN
        )

        result_crime.config(
            text="Crime: " + last_crime,
            fg=RED
        )

        timeline_label.config(
            text="Timeline: Idle → Detecting → Alert → Report"
        )


def update_camera():
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

    if len(faces) == 0:
        activity_label.config(
            text="No Activity Detected"
        )

    for x, y, w, h in faces:
        face = gray[
            y:y+h,
            x:x+w
        ]

        try:
            face = cv2.resize(
                face,
                (100, 100)
            )
        except:
            continue

        face = face.flatten().reshape(
            1,
            -1
        )

        prediction = model.predict(
            face
        )[0]

        person = names.get(
            prediction,
            {
                "name": "Unknown",
                "crime": "Unknown"
            }
        )

        person_name = person["name"]
        crime = person["crime"]

        update_detection(
            person_name,
            crime
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            3
        )

        label = person_name + " | " + crime

        cv2.rectangle(
            frame,
            (x, y-35),
            (x+w, y),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x+7, y-11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        frame
    )

    image.thumbnail(
        (800, 350)
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    camera_view.config(
        image=photo,
        text=""
    )

    camera_view.image = photo

    root.after(
        15,
        update_camera
    )


def update_detection(
    person_name,
    crime
):
    global last_person, last_crime

    last_person = person_name
    last_crime = crime

    status_label.config(
        text="SYSTEM ALERT – PERSON DETECTED",
        fg=RED
    )

    activity_label.config(
        text="Activity Detected"
    )

    result_title.config(
        text="DETECTION REPORT",
        fg=RED
    )

    result_person.config(
        text="Person: " + person_name,
        fg=GREEN
    )

    result_crime.config(
        text="Crime: " + crime,
        fg=RED
    )

    timeline_label.config(
        text="Timeline: Idle → Detecting → Alert → Report"
    )


def logout():
    stop_camera()

    dashboard.pack_forget()

    login_page.pack(
        fill="both",
        expand=True
    )

    username_entry.delete(
        0,
        tk.END
    )

    password_entry.delete(
        0,
        tk.END
    )


def exit_system():
    global camera

    if camera is not None:
        camera.release()

    root.destroy()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_alt.xml"
)


login_page = tk.Frame(
    root,
    bg=BG
)

login_left = tk.Frame(
    login_page,
    bg=BG
)

login_left.place(
    x=55,
    y=55,
    width=600,
    height=600
)

tk.Label(
    login_left,
    text="VISION",
    font=("Segoe UI", 44, "bold"),
    fg=WHITE,
    bg=BG
).pack(
    pady=(90, 0)
)

tk.Label(
    login_left,
    text="SHIELD",
    font=("Segoe UI", 44, "bold"),
    fg=BLUE,
    bg=BG
).pack()

tk.Label(
    login_left,
    text="AI-DRIVEN CRIMINAL DETECTION SYSTEM",
    font=("Segoe UI", 11, "bold"),
    fg=MUTED,
    bg=BG
).pack(
    pady=12
)

tk.Label(
    login_left,
    text="Instant Recognition • Real-Time Alerts\nContinuous Monitoring",
    font=("Segoe UI", 10),
    fg=BLUE,
    bg=BG,
    justify="center"
).pack(
    pady=20
)

tk.Label(
    login_left,
    text="● SECURE     ● REAL-TIME     ● AI POWERED",
    font=("Segoe UI", 9, "bold"),
    fg=GREEN,
    bg=BG
).pack(
    pady=18
)

login_card = tk.Frame(
    login_page,
    bg=PANEL,
    highlightbackground=BORDER,
    highlightthickness=1
)

login_card.place(
    x=700,
    y=55,
    width=470,
    height=610
)

tk.Label(
    login_card,
    text="Secure Access Portal",
    font=("Segoe UI", 24, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    pady=(55, 5)
)

tk.Label(
    login_card,
    text="Authenticate to unlock VisionShield",
    font=("Segoe UI", 10),
    fg=MUTED,
    bg=PANEL
).pack(
    pady=(0, 38)
)

tk.Label(
    login_card,
    text="USERNAME",
    font=("Segoe UI", 9, "bold"),
    fg=BLUE,
    bg=PANEL,
    anchor="w"
).pack(
    padx=50,
    fill="x"
)

username_entry = tk.Entry(
    login_card,
    font=("Segoe UI", 11),
    bg=DARK_PANEL,
    fg=WHITE,
    insertbackground=BLUE,
    relief="flat"
)

username_entry.pack(
    padx=50,
    pady=(7, 22),
    ipady=9,
    fill="x"
)

tk.Label(
    login_card,
    text="PASSWORD",
    font=("Segoe UI", 9, "bold"),
    fg=BLUE,
    bg=PANEL,
    anchor="w"
).pack(
    padx=50,
    fill="x"
)

password_entry = tk.Entry(
    login_card,
    font=("Segoe UI", 11),
    bg=DARK_PANEL,
    fg=WHITE,
    insertbackground=BLUE,
    show="*",
    relief="flat"
)

password_entry.pack(
    padx=50,
    pady=(7, 25),
    ipady=9,
    fill="x"
)

tk.Button(
    login_card,
    text="ACCESS SYSTEM  →",
    command=login,
    font=("Segoe UI", 10, "bold"),
    bg=BLUE,
    fg=BG,
    activebackground="#33CCFF",
    activeforeground=BG,
    relief="flat",
    cursor="hand2"
).pack(
    padx=50,
    fill="x",
    ipady=9
)

tk.Label(
    login_card,
    text="Restricted Access – Authorized Entry",
    font=("Segoe UI", 8, "bold"),
    fg=RED,
    bg=PANEL
).pack(
    pady=35
)

login_page.pack(
    fill="both",
    expand=True
)

dashboard = tk.Frame(
    root,
    bg=BG
)

header = tk.Frame(
    dashboard,
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
    text="AI-DRIVEN CRIMINAL DETECTION",
    font=("Segoe UI", 8, "bold"),
    fg=BLUE,
    bg=PANEL
).pack(
    side="left"
)

logout_button = tk.Button(
    header,
    text="LOGOUT",
    command=logout,
    font=("Segoe UI", 8, "bold"),
    bg=BLUE,
    fg=BG,
    activebackground="#33CCFF",
    relief="flat",
    cursor="hand2"
)

logout_button.pack(
    side="right",
    padx=20,
    ipadx=8,
    ipady=5
)

user_label = tk.Label(
    header,
    text="User: Sudhanshu  |  VisionShield",
    font=("Segoe UI", 8, "bold"),
    fg=TEXT,
    bg=PANEL
)

user_label.pack(
    side="right",
    padx=10
)

system_status = tk.Label(
    header,
    text="● SYSTEM ONLINE",
    font=("Segoe UI", 8, "bold"),
    fg=GREEN,
    bg=PANEL
)

system_status.pack(
    side="right",
    padx=10
)

main = tk.Frame(
    dashboard,
    bg=BG
)

main.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=18
)

watermark = tk.Label(
    dashboard,
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
    width=275,
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
    text="CONTROL CENTER",
    font=("Segoe UI", 13, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    pady=(25, 17)
)

camera_status = tk.Label(
    left_panel,
    text="● CAMERA INACTIVE",
    font=("Segoe UI", 8, "bold"),
    fg=RED,
    bg=PANEL
)

camera_status.pack(
    pady=(0, 15)
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
    padx=20,
    fill="x",
    ipady=7
)

stop_button = tk.Button(
    left_panel,
    text="■  STOP CAMERA",
    command=stop_camera,
    font=("Segoe UI", 9, "bold"),
    bg=BUTTON_GREY,
    fg=WHITE,
    activebackground=BUTTON_GREY_HOVER,
    activeforeground=WHITE,
    disabledforeground="#BDBDBD",
    relief="flat",
    cursor="hand2",
    state="disabled"
)

stop_button.pack(
    padx=20,
    pady=9,
    fill="x",
    ipady=7
)

exit_button = tk.Button(
    left_panel,
    text="🛡  EXIT SYSTEM",
    command=exit_system,
    font=("Segoe UI", 9, "bold"),
    bg=RED,
    fg=WHITE,
    activebackground="#FF3333",
    relief="flat",
    cursor="hand2"
)

exit_button.pack(
    padx=20,
    fill="x",
    ipady=7
)

tk.Label(
    left_panel,
    text="SYSTEM MODULES",
    font=("Segoe UI", 9, "bold"),
    fg=BLUE,
    bg=PANEL
).pack(
    pady=(27, 12)
)

modules = [
    "✓ Face Detection",
    "✓ KNN Recognition",
    "✓ OpenCV Processing",
    "✓ Live Camera",
    "✓ Face Recognition",
    "✓ Crime Identification",
    "✓ Real-Time Analysis"
]

for module in modules:
    tk.Label(
        left_panel,
        text=module,
        font=("Segoe UI", 8),
        fg=GREEN,
        bg=PANEL,
        anchor="w"
    ).pack(
        padx=28,
        pady=4,
        fill="x"
    )

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
    text="LIVE SURVEILLANCE",
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

result_card = tk.Frame(
    right_panel,
    bg=PANEL,
    height=120,
    highlightbackground=BORDER,
    highlightthickness=1
)

result_card.pack(
    fill="x",
    pady=(14, 0)
)

result_card.pack_propagate(False)

status_label = tk.Label(
    result_card,
    text="SYSTEM READY – AWAITING INPUT",
    font=("Segoe UI", 8, "bold"),
    fg=GREEN,
    bg=PANEL
)

status_label.place(
    x=20,
    y=12
)

activity_label = tk.Label(
    result_card,
    text="No Activity Detected",
    font=("Segoe UI", 8),
    fg=MUTED,
    bg=PANEL
)

activity_label.place(
    x=20,
    y=34
)

result_title = tk.Label(
    result_card,
    text="NO ACTIVITY DETECTED",
    font=("Segoe UI", 11, "bold"),
    fg=MUTED,
    bg=PANEL
)

result_title.place(
    x=20,
    y=56
)

result_person = tk.Label(
    result_card,
    text="Waiting for detection...",
    font=("Segoe UI", 14, "bold"),
    fg=TEXT,
    bg=PANEL
)

result_person.place(
    x=20,
    y=83
)

result_crime = tk.Label(
    result_card,
    text="Crime information will appear here",
    font=("Segoe UI", 9, "bold"),
    fg=MUTED,
    bg=PANEL
)

result_crime.place(
    x=320,
    y=78
)

timeline_label = tk.Label(
    result_card,
    text="Timeline: Idle",
    font=("Segoe UI", 8, "bold"),
    fg=BLUE,
    bg=PANEL
)

timeline_label.place(
    x=320,
    y=100
)

root.protocol(
    "WM_DELETE_WINDOW",
    exit_system
)

root.mainloop()