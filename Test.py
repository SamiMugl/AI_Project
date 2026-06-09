import face_recognition
import cv2
import time
from scipy.spatial import distance as dist
import playsound
from threading import Thread
import numpy as np
import random
import csv
import os

# Constants
MIN_AER = 0.30
THRESHOLD_LEFT = 50
THRESHOLD_RIGHT = 80
ALARM_DURATION = 3.0
BRAKE_TRIGGER = 6.0
SPEED_STEP = 5
MAX_SPEED = random.randint(75, 80)
MIN_SPEED = 0
LOG_FILE = "driver_logs.csv"

# Globals
ALARM_ON = False
ALARM_THREAD = None
ALARM_START = None
SIDE_START = None
BRAKE_ACTIVE = False
BRAKE_START = None
speed = MAX_SPEED

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Alert Type", "Current Speed"])

def log_alert(alert_type, current_speed):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, alert_type, f"{int(current_speed)} km/h"])

def sound_alarm(soundfile):
    global ALARM_ON
    while ALARM_ON:
        try:
            playsound.playsound(soundfile)
        except:
            break

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2 * C)

def main():
    global ALARM_ON, ALARM_THREAD, ALARM_START, SIDE_START, BRAKE_ACTIVE, BRAKE_START, speed

    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Driver Drowsiness Detection System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Driver Drowsiness Detection System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    blink_toggle = True

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        face_landmarks_list = face_recognition.face_landmarks(frame)

        ear = 1.0
        side_alarm = False
        sleep_alarm = False
        frame_center_x = w // 2

        if len(face_landmarks_list) > 0:
            for face_landmarks in face_landmarks_list:
                left_eye = face_landmarks['left_eye']
                right_eye = face_landmarks['right_eye']
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                cv2.polylines(frame, [np.array(left_eye)], True, (255, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_eye)], True, (255, 255, 0), 1)

                nose_bridge = face_landmarks['nose_bridge']
                nose_x = nose_bridge[0][0]

                if nose_x < (frame_center_x - THRESHOLD_LEFT) or nose_x > (frame_center_x + THRESHOLD_RIGHT):
                    if SIDE_START is None: SIDE_START = time.time()
                    elif time.time() - SIDE_START >= ALARM_DURATION: side_alarm = True
                else:
                    SIDE_START = None

                if ear < MIN_AER:
                    if ALARM_START is None: ALARM_START = time.time()
                    elif time.time() - ALARM_START >= ALARM_DURATION: sleep_alarm = True
                else:
                    ALARM_START = None
        else:
            if SIDE_START is None: SIDE_START = time.time()
            elif time.time() - SIDE_START >= ALARM_DURATION: side_alarm = True

        alarm_active = side_alarm or sleep_alarm

        if alarm_active:
            if not ALARM_ON:
                ALARM_ON = True
                reason = "Sleeping" if sleep_alarm else "Distraction"
                log_alert(reason, speed)
                if ALARM_THREAD is None or not ALARM_THREAD.is_alive():
                    ALARM_THREAD = Thread(target=sound_alarm, args=('alarm_wav.mp3',))
                    ALARM_THREAD.daemon = True
                    ALARM_THREAD.start()

            if BRAKE_START is None:
                BRAKE_START = time.time()
            elif time.time() - BRAKE_START >= BRAKE_TRIGGER:
                BRAKE_ACTIVE = True
        else:
            ALARM_ON = False
            BRAKE_START = None

        if BRAKE_ACTIVE:
            if speed > MIN_SPEED: speed = max(MIN_SPEED, speed - SPEED_STEP)
            if blink_toggle:
                # Braking text ko bhi dynamic center kiya hai
                b_text = "Applying Brakes"
                (b_w, b_h), _ = cv2.getTextSize(b_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
                cv2.putText(frame, b_text, ((w - b_w) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
            blink_toggle = not blink_toggle
            cv2.circle(frame, (50, h // 2), 20, (0, 0, 255), -1)
            cv2.circle(frame, (w - 50, h // 2), 20, (0, 0, 255), -1)
            if speed <= MIN_SPEED: BRAKE_ACTIVE = False
        else:
            if speed < MAX_SPEED: speed = min(MAX_SPEED, speed + SPEED_STEP)

        # --- HORIZONTAL CENTERING FOR SPEED ---
        speed_text = f"Speed: {int(speed)} km/h"
        (s_w, s_h), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(frame, speed_text, ((w - s_w) // 2, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        if sleep_alarm: cv2.putText(frame, "ALERT! Sleeping!", (w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif side_alarm: cv2.putText(frame, "ALERT! Be Focused!", (w - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Driver Drowsiness Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ALARM_ON = False
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()