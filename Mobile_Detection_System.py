# Mobile Detection System
# Detects cell phone usage while driving via webcam (YOLOv8).
# Install: pip install ultralytics opencv-python playsound

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
from ultralytics import YOLO

# Constants
DETECTION_DURATION = 5.0
CONFIDENCE = 0.30
IOU_THRESHOLD = 0.45
CELL_PHONE_CLASS = 67
# Keep counting if phone/driver briefly lost while moving (YOLO flicker)
PHONE_GRACE_SEC = 2.0
DRIVER_GRACE_SEC = 2.0
SESSION_RESET_SEC = 2.5
SMOOTH_WINDOW = 8
SMOOTH_MIN_HITS = 2
LOG_FILE = "mobile_logs.csv"
WINDOW_NAME = "Mobile Detection System"
ALARM_FILE = "alarm_wav.mp3"
MODEL_NAME = "yolov8n.pt"
FACE_MIN_CONFIDENCE = 0.65
FACE_MIN_SIZE = 70

SPEED_STEP = 5
SPEED_CHANGE_INTERVAL = 1.0
BRAKE_TRIGGER = 6.0
MAX_SPEED = random.randint(75, 80)
MIN_SPEED = 0

# Globals
ALARM_ON = False
ALARM_THREAD = None
MOBILE_START = None
BRAKE_ACTIVE = False
BRAKE_START = None
speed = MAX_SPEED
last_speed_update = None
last_log_time = 0.0
LOG_COOLDOWN = 2.0

model = None
face_detector = None
phone_hit_history = deque(maxlen=SMOOTH_WINDOW)
last_phone_seen = None
last_driver_seen = None
last_phone_detections = []
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
            playsound.playsound(soundfile)
        except Exception:
            break


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


def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_NAME)
    return model


def get_face_detector():
    global face_detector
    if face_detector is None:
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=FACE_MIN_CONFIDENCE,
        )
    return face_detector


def box_overlap_ratio(face_box, obj_box):
    """Intersection over face area — high value means object covers the 'face' box."""
    ft, fr, fb, fl = face_box
    x1, y1, x2, y2 = obj_box[:4]
    ix1, iy1 = max(fl, x1), max(ft, y1)
    ix2, iy2 = min(fr, x2), min(fb, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    face_area = max((fr - fl) * (fb - ft), 1)
    return inter / face_area


def is_valid_face_box(left, top, width, height, frame_w, frame_h):
    if width < FACE_MIN_SIZE or height < FACE_MIN_SIZE:
        return False
    aspect = width / max(height, 1)
    if aspect < 0.65 or aspect > 1.55:
        return False
    if top > frame_h * 0.88:
        return False
    return True


def detect_driver(frame, phone_detections=None):
    """Detect real face only (MediaPipe — avoids hand false positives)."""
    if frame is None or frame.size == 0:
        return False, []

    frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = get_face_detector().process(rgb)

    locations = []
    if not results.detections:
        return False, locations

    for det in results.detections:
        if det.score[0] < FACE_MIN_CONFIDENCE:
            continue

        bbox = det.location_data.relative_bounding_box
        left = int(bbox.xmin * w)
        top = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        if not is_valid_face_box(left, top, width, height, w, h):
            continue

        right = min(left + width, w)
        bottom = min(top + height, h)
        left = max(left, 0)
        top = max(top, 0)
        face_box = (top, right, bottom, left)

        if phone_detections:
            on_phone = any(
                box_overlap_ratio(face_box, phone) > 0.35 for phone in phone_detections
            )
            if on_phone:
                continue

        locations.append(face_box)

    if len(locations) > 1:
        locations = [
            max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        ]

    return len(locations) > 0, locations


def detect_mobile(frame, yolo_model):
    results = yolo_model(
        frame,
        verbose=False,
        classes=[CELL_PHONE_CLASS],
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        imgsz=640,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            detections.append((x1, y1, x2, y2, score))
    return detections


def draw_face_boxes(frame, locations):
    for top, right, bottom, left in locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 200, 0), 2)
        cv2.putText(
            frame,
            "Driver",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )


def draw_phone_boxes(frame, detections, alarm_active, tracking=False):
    color = (0, 0, 255) if alarm_active else (0, 255, 0)
    if tracking and not detections:
        color = (0, 165, 255)
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "Mobile (tracking)" if tracking else f"Mobile {score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def phone_recently_seen(now):
    return last_phone_seen is not None and (now - last_phone_seen) <= PHONE_GRACE_SEC


def driver_recently_seen(now):
    return last_driver_seen is not None and (now - last_driver_seen) <= DRIVER_GRACE_SEC


def smooth_phone_visible(raw_visible):
    phone_hit_history.append(1 if raw_visible else 0)
    if len(phone_hit_history) < 3:
        return raw_visible
    return sum(phone_hit_history) >= SMOOTH_MIN_HITS


def is_driver_using_phone(now, raw_phone, raw_driver):
    phone_ok = raw_phone or phone_recently_seen(now)
    driver_ok = raw_driver or driver_recently_seen(now)
    return phone_ok and driver_ok


def main():
    global ALARM_ON, ALARM_THREAD, MOBILE_START, BRAKE_ACTIVE, BRAKE_START, speed, last_log_time
    global last_speed_update
    global last_phone_seen, last_driver_seen, last_phone_detections

    yolo_model = get_model()
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    blink_toggle = True

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        now = time.time()

        phone_detections = detect_mobile(frame, yolo_model)
        driver_present, face_locations = detect_driver(frame, phone_detections)

        raw_phone = len(phone_detections) > 0
        phone_visible = smooth_phone_visible(raw_phone)

        if raw_phone:
            last_phone_seen = now
            last_phone_detections = phone_detections
        if driver_present:
            last_driver_seen = now

        driver_using_phone = is_driver_using_phone(now, phone_visible, driver_present)
        phone_tracking = driver_using_phone and not raw_phone

        mobile_alarm = False
        hold_seconds = 0.0

        if driver_using_phone:
            if MOBILE_START is None:
                MOBILE_START = now
            hold_seconds = now - MOBILE_START
            if hold_seconds >= DETECTION_DURATION:
                mobile_alarm = True
        else:
            mobile_alarm = False
            MOBILE_START = None
            hold_seconds = 0.0

        if not raw_phone:
            mobile_alarm = False

        if driver_present:
            draw_face_boxes(frame, face_locations)

        boxes_to_draw = phone_detections if raw_phone else (
            last_phone_detections if phone_tracking else []
        )
        draw_phone_boxes(frame, boxes_to_draw, mobile_alarm, tracking=phone_tracking)

        if mobile_alarm:
            if not ALARM_ON:
                ALARM_ON = True
                if now - last_log_time >= LOG_COOLDOWN:
                    log_alert("Mobile Use", speed)
                    last_log_time = now
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
            BRAKE_ACTIVE = False
            last_speed_update = now - SPEED_CHANGE_INTERVAL

        if BRAKE_ACTIVE:
            update_speed_gradual(now, braking=True)
            if blink_toggle:
                b_text = "Applying Brakes"
                (b_w, _), _ = cv2.getTextSize(b_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
                cv2.putText(
                    frame,
                    b_text,
                    ((w - b_w) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    3,
                )
            blink_toggle = not blink_toggle
            cv2.circle(frame, (50, h // 2), 20, (0, 0, 255), -1)
            cv2.circle(frame, (w - 50, h // 2), 20, (0, 0, 255), -1)
        else:
            update_speed_gradual(now, braking=False)

        speed_text = f"Speed: {int(speed)} km/h"
        (s_w, _), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(
            frame,
            speed_text,
            ((w - s_w) // 2, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )

        status = "Driver: OK" if driver_present else "Driver: Not Visible"
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

        if phone_tracking:
            cv2.putText(
                frame,
                "Mobile tracking (movement OK)",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )

        if phone_visible and not driver_present and not driver_recently_seen(now):
            cv2.putText(
                frame,
                "Mobile seen - waiting for driver",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
        elif driver_using_phone and not mobile_alarm:
            cv2.putText(
                frame,
                f"Mobile hold: {hold_seconds:.1f}s / {DETECTION_DURATION:.0f}s",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
        elif mobile_alarm:
            cv2.putText(
                frame,
                "ALERT! Stop Using Mobile!",
                (w - 420, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ALARM_ON = False
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()