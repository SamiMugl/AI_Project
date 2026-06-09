# Driver Drowsiness Detection System
# Detects 5 risky behaviors: eyes closed, look left/right/up/down

import csv
import os
import random
import time
from collections import deque
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np
import playsound
from scipy.spatial import distance as dist

# --- Constants ---
MIN_EAR_FLOOR = 0.18
EAR_CLOSED_RATIO = 0.50
ALARM_DURATION = 5.0
BRAKE_TRIGGER = 6.0
SPEED_STEP = 5
SPEED_CHANGE_INTERVAL = 1.0
MAX_SPEED = random.randint(75, 80)
MIN_SPEED = 0
LOG_FILE = "driver_logs.csv"
WINDOW_NAME = "Driver Drowsiness Detection System"
ALARM_FILE = "alarm_wav.mp3"

YAW_LEFT = -0.28
YAW_RIGHT = 0.28
PITCH_UP = -0.12
PITCH_DOWN = 0.12

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1
CHIN_IDX = 152
FOREHEAD_IDX = 10
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX = 454

SMOOTH_FRAMES = 8
SMOOTH_HITS = 5
SLEEP_SMOOTH_HITS = 4
EAR_SMOOTH = 4
POSE_SMOOTH = 6
NEUTRAL_SAMPLES = 40
YAW_NEUTRAL_LIMIT = 0.15
PITCH_NEUTRAL_LOW = 0.42
PITCH_NEUTRAL_HIGH = 0.58
FACE_LOST_GRACE = 0.4

RISK_LABELS = {
    "sleep": ("Sleeping", "ALERT! Eyes Closed!"),
    "left": ("Look Left", "ALERT! Looking Left!"),
    "right": ("Look Right", "ALERT! Looking Right!"),
    "up": ("Look Up", "ALERT! Looking Up!"),
    "down": ("Look Down", "ALERT! Looking Down!"),
}

# --- Globals ---
ALARM_ON = False
ALARM_THREAD = None
BRAKE_ACTIVE = False
BRAKE_START = None
speed = MAX_SPEED
last_speed_update = None
face_mesh = None
risk_timers = {key: None for key in RISK_LABELS}
risk_smooth = {key: deque(maxlen=SMOOTH_FRAMES) for key in RISK_LABELS}
ear_history = deque(maxlen=EAR_SMOOTH)
yaw_history = deque(maxlen=POSE_SMOOTH)
pitch_history = deque(maxlen=POSE_SMOOTH)
baseline_yaw_samples = deque(maxlen=NEUTRAL_SAMPLES)
baseline_pitch_samples = deque(maxlen=NEUTRAL_SAMPLES)
neutral_yaw = 0.0
neutral_pitch = 0.5
face_lost_since = None
baseline_ear_samples = deque(maxlen=45)
active_alarm_key = None
last_logged_key = None
ear_threshold = MIN_EAR_FLOOR

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Alert Type", "Current Speed"])


def log_alert(alert_type, current_speed):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, alert_type, f"{int(current_speed)} km/h"])


def sound_alarm(soundfile):
    global ALARM_ON
    while ALARM_ON:
        try:
            playsound.playsound(soundfile, block=True)
        except Exception:
            break


def get_face_mesh():
    global face_mesh
    if face_mesh is None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.65,
        )
    return face_mesh


def landmark_xy(landmarks, index, width, height):
    pt = landmarks.landmark[index]
    return int(pt.x * width), int(pt.y * height)


def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 1.0
    return (A + B) / (2.0 * C)


def get_eye_points(landmarks, indices, width, height):
    return [landmark_xy(landmarks, i, width, height) for i in indices]


def compute_eyes(landmarks, width, height):
    left_pts = get_eye_points(landmarks, LEFT_EYE_IDX, width, height)
    right_pts = get_eye_points(landmarks, RIGHT_EYE_IDX, width, height)
    left_ear = eye_aspect_ratio(left_pts)
    right_ear = eye_aspect_ratio(right_pts)
    avg_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, avg_ear, left_pts, right_pts


def get_ear_threshold():
    if len(baseline_ear_samples) < 12:
        return MIN_EAR_FLOOR
    baseline = sum(baseline_ear_samples) / len(baseline_ear_samples)
    return max(MIN_EAR_FLOOR, baseline * EAR_CLOSED_RATIO)


def is_eyes_closed(left_ear, right_ear, threshold):
    return left_ear < threshold and right_ear < threshold


def update_ear_baseline(avg_ear, rel_yaw, rel_pitch):
    if abs(rel_yaw) > YAW_NEUTRAL_LIMIT:
        return
    if abs(rel_pitch) > 0.08:
        return
    if avg_ear > MIN_EAR_FLOOR:
        baseline_ear_samples.append(avg_ear)


def draw_eye(frame, eye_points):
    cv2.polylines(
        frame,
        [np.array(eye_points, dtype=np.int32)],
        True,
        (255, 255, 0),
        1,
    )


def get_head_pose(landmarks, width, height):
    nose_x, nose_y = landmark_xy(landmarks, NOSE_TIP_IDX, width, height)
    _, chin_y = landmark_xy(landmarks, CHIN_IDX, width, height)
    _, forehead_y = landmark_xy(landmarks, FOREHEAD_IDX, width, height)
    lx, _ = landmark_xy(landmarks, LEFT_CHEEK_IDX, width, height)
    rx, _ = landmark_xy(landmarks, RIGHT_CHEEK_IDX, width, height)

    face_cx = (lx + rx) / 2.0
    face_w = max(rx - lx, 1)
    face_h = max(chin_y - forehead_y, 1)

    yaw = (nose_x - face_cx) / (face_w * 0.5)
    pitch = (nose_y - forehead_y) / face_h
    return yaw, pitch


def smooth_pose(yaw, pitch):
    yaw_history.append(yaw)
    pitch_history.append(pitch)
    return sum(yaw_history) / len(yaw_history), sum(pitch_history) / len(pitch_history)


def update_neutral_pose(yaw, pitch, eyes_open):
    global neutral_yaw, neutral_pitch
    if not eyes_open:
        return
    if abs(yaw - neutral_yaw) > YAW_NEUTRAL_LIMIT:
        return
    if pitch < PITCH_NEUTRAL_LOW or pitch > PITCH_NEUTRAL_HIGH:
        return
    baseline_yaw_samples.append(yaw)
    baseline_pitch_samples.append(pitch)
    if len(baseline_yaw_samples) >= 15:
        neutral_yaw = sum(baseline_yaw_samples) / len(baseline_yaw_samples)
    if len(baseline_pitch_samples) >= 15:
        neutral_pitch = sum(baseline_pitch_samples) / len(baseline_pitch_samples)


def pick_head_direction(rel_yaw, rel_pitch):
    """Only one direction at a time — strongest deviation wins."""
    candidates = []
    if rel_yaw < YAW_LEFT:
        candidates.append(("left", YAW_LEFT - rel_yaw))
    if rel_yaw > YAW_RIGHT:
        candidates.append(("right", rel_yaw - YAW_RIGHT))
    if rel_pitch < PITCH_UP:
        candidates.append(("up", PITCH_UP - rel_pitch))
    if rel_pitch > PITCH_DOWN:
        candidates.append(("down", rel_pitch - PITCH_DOWN))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def smooth_flag(key, raw_active, hits_needed=None):
    if hits_needed is None:
        hits_needed = SMOOTH_HITS
    risk_smooth[key].append(1 if raw_active else 0)
    if len(risk_smooth[key]) < hits_needed:
        return False
    return sum(risk_smooth[key]) >= hits_needed


def detect_risks(landmarks, width, height, left_ear, right_ear, threshold):
    raw_yaw, raw_pitch = get_head_pose(landmarks, width, height)
    yaw, pitch = smooth_pose(raw_yaw, raw_pitch)

    rel_yaw = yaw - neutral_yaw
    rel_pitch = pitch - neutral_pitch
    eyes_open = not is_eyes_closed(left_ear, right_ear, threshold)
    update_neutral_pose(yaw, pitch, eyes_open)

    head_dir = pick_head_direction(rel_yaw, rel_pitch)
    raw = {
        "sleep": is_eyes_closed(left_ear, right_ear, threshold),
        "left": head_dir == "left",
        "right": head_dir == "right",
        "up": head_dir == "up",
        "down": head_dir == "down",
    }

    stable = {}
    for key, val in raw.items():
        hits = SLEEP_SMOOTH_HITS if key == "sleep" else SMOOTH_HITS
        stable[key] = smooth_flag(key, val, hits)
    return stable, rel_yaw, rel_pitch, raw


def is_focused(stable, left_ear, right_ear, threshold, face_ok):
    if not face_ok:
        return False
    if any(stable.values()):
        return False
    if is_eyes_closed(left_ear, right_ear, threshold):
        return False
    return True


def reset_all():
    global ALARM_ON, BRAKE_START, BRAKE_ACTIVE, active_alarm_key, last_logged_key
    for key in risk_timers:
        risk_timers[key] = None
    ALARM_ON = False
    BRAKE_START = None
    BRAKE_ACTIVE = False
    active_alarm_key = None
    last_logged_key = None


def update_speed_gradual(now, braking):
    """Change speed by 5 km/h at a time, not every frame."""
    global speed, last_speed_update

    if last_speed_update is None:
        last_speed_update = now
        return

    if now - last_speed_update < SPEED_CHANGE_INTERVAL:
        return

    last_speed_update = now
    if braking:
        if speed > MIN_SPEED:
            speed = max(MIN_SPEED, speed - SPEED_STEP)
    elif speed < MAX_SPEED:
        speed = min(MAX_SPEED, speed + SPEED_STEP)


def update_timers(stable, now):
    global active_alarm_key
    priority = ["sleep", "down", "up", "left", "right"]
    triggered = None

    for key in priority:
        if stable[key]:
            if risk_timers[key] is None:
                risk_timers[key] = now
            elif (now - risk_timers[key]) >= ALARM_DURATION:
                triggered = key
        else:
            risk_timers[key] = None

    active_alarm_key = triggered
    return active_alarm_key


def pick_display_risk(stable, alarm_key):
    if alarm_key:
        return alarm_key
    for key in ["sleep", "down", "up", "left", "right"]:
        if stable.get(key):
            return key
    return None


def main():
    global ALARM_ON, ALARM_THREAD, BRAKE_ACTIVE, BRAKE_START, speed, last_speed_update
    global active_alarm_key, last_logged_key, ear_threshold, face_lost_since

    mesh = get_face_mesh()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    blink_toggle = True

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
        h, w = frame.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        ear = 1.0
        left_ear = 1.0
        right_ear = 1.0
        rel_yaw = 0.0
        rel_pitch = 0.0
        face_ok = False
        stable = {k: False for k in RISK_LABELS}
        alarm_key = None

        if results.multi_face_landmarks:
            face_ok = True
            face_lost_since = None
            lm = results.multi_face_landmarks[0]
            raw_left, raw_right, raw_avg, left_eye, right_eye = compute_eyes(lm, w, h)
            ear_history.append(raw_avg)
            ear = sum(ear_history) / len(ear_history)
            left_ear = raw_left
            right_ear = raw_right

            draw_eye(frame, left_eye)
            draw_eye(frame, right_eye)

            ear_threshold = get_ear_threshold()
            stable, rel_yaw, rel_pitch, _ = detect_risks(
                lm, w, h, left_ear, right_ear, ear_threshold
            )
            update_ear_baseline(ear, rel_yaw, rel_pitch)

            if is_focused(stable, left_ear, right_ear, ear_threshold, face_ok):
                reset_all()
            else:
                alarm_key = update_timers(stable, now)
        else:
            if face_lost_since is None:
                face_lost_since = now
            elif now - face_lost_since >= FACE_LOST_GRACE:
                reset_all()
                for dq in risk_smooth.values():
                    dq.clear()
                yaw_history.clear()
                pitch_history.clear()

        alarm_active = alarm_key is not None

        if alarm_active:
            if not ALARM_ON:
                ALARM_ON = True
                log_name = RISK_LABELS[alarm_key][0]
                if last_logged_key != alarm_key:
                    log_alert(log_name, speed)
                    last_logged_key = alarm_key
                if ALARM_THREAD is None or not ALARM_THREAD.is_alive():
                    ALARM_THREAD = Thread(target=sound_alarm, args=(ALARM_FILE,))
                    ALARM_THREAD.daemon = True
                    ALARM_THREAD.start()

            if BRAKE_START is None:
                BRAKE_START = now
            elif now - BRAKE_START >= BRAKE_TRIGGER:
                BRAKE_ACTIVE = True
        else:
            ALARM_ON = False
            BRAKE_START = None

        if BRAKE_ACTIVE:
            update_speed_gradual(now, braking=True)
            if blink_toggle:
                txt = "Applying Brakes"
                (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
                cv2.putText(
                    frame, txt, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3,
                )
            blink_toggle = not blink_toggle
            cv2.circle(frame, (50, h // 2), 20, (0, 0, 255), -1)
            cv2.circle(frame, (w - 50, h // 2), 20, (0, 0, 255), -1)
            if speed <= MIN_SPEED:
                BRAKE_ACTIVE = False
        else:
            update_speed_gradual(now, braking=False)

        speed_text = f"Speed: {int(speed)} km/h"
        (sw, _), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(
            frame, speed_text, ((w - sw) // 2, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3,
        )

        cv2.putText(
            frame, "Driver: OK" if face_ok else "Driver: Not Visible",
            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2,
        )
        cv2.putText(
            frame,
            f"EAR: {ear:.2f} (sleep below {ear_threshold:.2f})  Head: {rel_yaw:+.2f} / {rel_pitch:+.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        if len(baseline_yaw_samples) < 15:
            cv2.putText(
                frame,
                "Calibrating - look straight at camera for 2 sec",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
            )

        show_key = pick_display_risk(stable, alarm_key)
        if show_key:
            label = RISK_LABELS[show_key][0]
            timer = risk_timers.get(show_key)
            if timer and not alarm_key:
                left = max(0.0, ALARM_DURATION - (now - timer))
                status = f"Detecting: {label} ({left:.1f}s)"
            else:
                status = RISK_LABELS[show_key][1]
            cv2.putText(
                frame, status, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2,
            )
        else:
            cv2.putText(
                frame, "Focus: OK - Keep eyes on road",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
            )

        cv2.putText(
            frame, "Press Q to quit", (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ALARM_ON = False
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
