# Dashboard

import customtkinter as ctk
import threading
import Driver_Drowsiness_Detector as drowsiness
import Mobile_Detection_System as mobile_detection
import Virtual_Volume_Changer as volume_controller


# Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# Functions
def run_driver_detection():
    thread = threading.Thread(target=drowsiness.main)
    thread.daemon = True
    thread.start()


def run_volume_controller():
    thread = threading.Thread(target=volume_controller.main)
    thread.daemon = True
    thread.start()


def run_mobile_detection():
    thread = threading.Thread(target=mobile_detection.main)
    thread.daemon = True
    thread.start()


# Main Window
root = ctk.CTk()
root.title("Detection Systems Dashboard")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

bg_color = "#081426"

root.geometry(f"{screen_width}x{screen_height-70}+0+0")
root.configure(fg_color=bg_color)


# Side Design
left_outer = ctk.CTkFrame(root, width=4, fg_color="#00eaff")
left_outer.place(relx=0.05, rely=0.2, relheight=0.6)

left_top = ctk.CTkFrame(root, height=4, width=50, fg_color="#00eaff")
left_top.place(relx=0.05, rely=0.2)

left_bottom = ctk.CTkFrame(root, height=4, width=50, fg_color="#00eaff")
left_bottom.place(relx=0.05, rely=0.8)

left_inner = ctk.CTkFrame(root, width=3, fg_color="#00eaff")
left_inner.place(relx=0.075, rely=0.3, relheight=0.4)

right_outer = ctk.CTkFrame(root, width=4, fg_color="#00eaff")
right_outer.place(relx=0.95, rely=0.2, relheight=0.6)

right_top = ctk.CTkFrame(root, height=4, width=50, fg_color="#00eaff")
right_top.place(relx=0.914, rely=0.2)

right_bottom = ctk.CTkFrame(root, height=4, width=50, fg_color="#00eaff")
right_bottom.place(relx=0.914, rely=0.8)

right_inner = ctk.CTkFrame(root, width=3, fg_color="#00eaff")
right_inner.place(relx=0.93, rely=0.3, relheight=0.4)


# Heading
title = ctk.CTkLabel(
    root,
    text="🤖 AI Detection Systems 🤖",
    font=("Arial", 36, "bold"),
    text_color="#00eaff"
)
title.pack(pady=40)

subtitle = ctk.CTkLabel(
    root,
    text="Choose Your Detection Module",
    font=("Arial", 16),
    text_color="#cbd5f5"
)
subtitle.pack()

line = ctk.CTkFrame(root, height=2, width=450, fg_color="#00eaff")
line.pack(pady=10)


# Frame
frame = ctk.CTkFrame(root, fg_color=bg_color)
frame.pack(expand=True, pady=60)


# Buttons
btn1 = ctk.CTkButton(
    frame,
    text="Driver Drowsiness Detection",
    font=("Arial", 18, "bold"),
    width=420,
    height=55,
    corner_radius=15,
    fg_color="#ff5733",
    hover_color="#ff7b5a",
    border_width=2,
    border_color="white",
    command=run_driver_detection
)
btn1.pack(pady=15)

btn2 = ctk.CTkButton(
    frame,
    text="Virtual Volume Controller",
    font=("Arial", 18, "bold"),
    width=420,
    height=55,
    corner_radius=15,
    fg_color="#3498db",
    hover_color="#5dade2",
    border_width=2,
    border_color="white",
    command=run_volume_controller
)
btn2.pack(pady=15)

btn3 = ctk.CTkButton(
    frame,
    text="Mobile Detection System",
    font=("Arial", 18, "bold"),
    width=420,
    height=55,
    corner_radius=15,
    fg_color="#2ecc71",
    hover_color="#58d68d",
    border_width=2,
    border_color="white",
    command=run_mobile_detection,
)
btn3.pack(pady=15)


# Exit Button
exit_btn = ctk.CTkButton(
    root,
    text="Exit",
    font=("Arial", 14, "bold"),
    width=110,
    height=42,
    corner_radius=10,
    fg_color="red",
    hover_color="#ff4d4d",
    command=root.destroy
)

exit_btn.place(x=30, y=screen_height-130)


# Main Loop
root.mainloop()