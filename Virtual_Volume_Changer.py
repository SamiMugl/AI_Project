# Virtual Volume Controller
# Hand-gesture volume + soft demo music for presentation.

import math
import os
import struct
import time
import wave
from collections import deque
from threading import Thread

import cv2
import numpy as np
import playsound
from comtypes import CLSCTX_ALL, GUID
from ctypes import POINTER, cast
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import Hand_Tracking_Module as htm

try:
    import pygame
except ImportError:
    pygame = None

WINDOW_NAME = "Virtual Volume Controller"
WIDTH_CAM, HEIGHT_CAM = 1280, 720
PINCH_MIN, PINCH_MAX = 50, 200
# Apni music project folder mein — pehli mili file use hogi
MUSIC_CANDIDATES = [
    "Sakura-Girl-Sweet-Memories-chosic.com_.mp3",
    "background_music.mp3",
    "background_music.wav",
    "background_music.ogg",
    "music.mp3",
    "music.wav",
]
FALLBACK_MUSIC_FILE = "demo_music.wav"
DEMO_START_PERCENT = 55
SMOOTH_WINDOW = 4

CYAN = (255, 234, 0)
ORANGE = (50, 120, 255)
DARK = (32, 18, 12)
GREEN = (120, 255, 0)

SIDEBAR_W = 190
TOP_H = 70
BOTTOM_H = 75
BG_COLOR = (12, 18, 32)

MUSIC_PLAYING = False
MUSIC_THREAD = None


def resource_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def create_soft_demo_music(path, duration_sec=20, sample_rate=44100):
    notes = [261.63, 329.63, 392.0, 493.88]
    pack = struct.Struct("<h")
    total = duration_sec * sample_rate

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(total):
            t = i / sample_rate
            sample = sum(
                0.06 * math.sin(2 * math.pi * f * t + n * 0.4)
                for n, f in enumerate(notes)
            )
            sample *= 0.6 + 0.4 * math.sin(2 * math.pi * 0.15 * t)
            wf.writeframes(pack.pack(int(max(-32768, min(32767, sample * 32767)))))


def get_music_path():
    """User file first, else auto-generated fallback."""
    for name in MUSIC_CANDIDATES:
        path = resource_path(name)
        if os.path.isfile(path):
            return path, name
    fallback = resource_path(FALLBACK_MUSIC_FILE)
    if not os.path.isfile(fallback):
        create_soft_demo_music(fallback)
    return fallback, FALLBACK_MUSIC_FILE


def _playsound_loop(path):
    global MUSIC_PLAYING
    while MUSIC_PLAYING:
        try:
            playsound.playsound(path, block=True)
        except Exception:
            break


def start_demo_music():
    global MUSIC_PLAYING, MUSIC_THREAD
    path, music_name = get_music_path()
    MUSIC_PLAYING = True

    if pygame is not None:
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(0.45)
            pygame.mixer.music.play(-1)
            time.sleep(0.15)
            if pygame.mixer.music.get_busy():
                return f"Playing: {music_name}"
        except Exception:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    MUSIC_THREAD = Thread(target=_playsound_loop, args=(path,), daemon=True)
    MUSIC_THREAD.start()
    time.sleep(0.2)
    return f"Playing: {music_name}"


def stop_demo_music():
    global MUSIC_PLAYING, MUSIC_THREAD
    MUSIC_PLAYING = False
    if pygame is not None:
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass
    MUSIC_THREAD = None


def get_system_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        GUID(IAudioEndpointVolume._iid_), CLSCTX_ALL, None
    )
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = volume.GetVolumeRange()
    return volume, vol_range[0], vol_range[1]


def percent_to_db(percent, min_db, max_db):
    """0-100% ko natural loudness ke sath dB mein (linear dB se zyada quiet nahi)."""
    p = max(0.0, min(100.0, percent)) / 100.0
    if p <= 0.0:
        return float(min_db)
    if p >= 1.0:
        return float(max_db)
    target = 20.0 * math.log10(p)
    return float(max(min_db, min(max_db, target)))


def db_to_percent(level, min_db, max_db):
    if level >= max_db:
        return 100
    if level <= min_db:
        return 0
    amp = 10.0 ** (level / 20.0)
    return int(max(0, min(100, round(amp * 100.0))))


def set_listen_volume(volume, min_vol, max_vol, percent=55):
    level = percent_to_db(percent, min_vol, max_vol)
    volume.SetMasterVolumeLevel(level, None)
    return level


def smooth_value(history, value):
    history.append(value)
    return sum(history) / len(history)


def draw_top_bar(img, fps):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 70), DARK, -1)
    cv2.line(img, (0, 70), (w, 70), CYAN, 2)
    cv2.putText(
        img,
        "VIRTUAL VOLUME CONTROLLER",
        (20, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        CYAN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Pinch thumb + index  |  Q = quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"FPS {int(fps)}",
        (w - 110, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (100, 100, 255),
        2,
        cv2.LINE_AA,
    )


def build_layout(cam_frame):
    """Camera on the right, empty sidebar on the left — no overlap."""
    ch, cw = cam_frame.shape[:2]
    canvas_h = ch + TOP_H + BOTTOM_H
    canvas_w = SIDEBAR_W + cw
    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)
    canvas[TOP_H : TOP_H + ch, SIDEBAR_W : SIDEBAR_W + cw] = cam_frame
    return canvas


def draw_volume_bar(canvas, vol_pct):
    h, w = canvas.shape[:2]
    x, bar_w = 28, 44
    y1, y2 = TOP_H + 30, h - BOTTOM_H - 30
    bar_h = y2 - y1
    panel_w = SIDEBAR_W - 10

    cv2.rectangle(canvas, (8, y1 - 28), (8 + panel_w, y2 + 40), DARK, -1)
    cv2.rectangle(canvas, (8, y1 - 28), (8 + panel_w, y2 + 40), CYAN, 1)

    cv2.putText(
        canvas, "VOL", (x + bar_w + 8, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2, cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (x, y1), (x + bar_w, y2), (70, 70, 70), 2)

    fill_top = int(y2 - (bar_h * vol_pct / 100.0))
    cv2.rectangle(canvas, (x + 3, fill_top), (x + bar_w - 3, y2 - 3), ORANGE, cv2.FILLED)

    cv2.putText(
        canvas, f"{int(vol_pct)}%", (x + bar_w + 8, y1 + bar_h // 2 + 8),
        cv2.FONT_HERSHEY_DUPLEX, 0.85, ORANGE, 1, cv2.LINE_AA,
    )

    cv2.line(canvas, (SIDEBAR_W - 2, TOP_H), (SIDEBAR_W - 2, h - BOTTOM_H), CYAN, 1)


def draw_status(canvas, hand_ok, music_status):
    h, w = canvas.shape[:2]
    y = h - 52
    cv2.rectangle(canvas, (0, h - BOTTOM_H), (w, h), DARK, -1)
    cv2.line(canvas, (0, h - BOTTOM_H), (w, h - BOTTOM_H), CYAN, 1)

    hand_txt = "Hand detected - pinch to change volume" if hand_ok else "Show hand to camera"
    hand_col = GREEN if hand_ok else (0, 200, 255)
    cv2.putText(
        canvas, hand_txt, (SIDEBAR_W + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, hand_col, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"Music: {music_status}", (SIDEBAR_W + 20, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA,
    )


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_CAM)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_CAM)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    detector = htm.HandDetector(detectionCon=0.75, maxHands=1)
    volume, min_vol, max_vol = get_system_volume()

    set_listen_volume(volume, min_vol, max_vol, percent=DEMO_START_PERCENT)
    music_status = start_demo_music()

    vol_history = deque(maxlen=SMOOTH_WINDOW)
    vol_pct = db_to_percent(volume.GetMasterVolumeLevel(), min_vol, max_vol)
    prev_time = time.time()

    while True:
        ok, img = cap.read()
        if not ok or img is None:
            continue

        img = cv2.flip(img, 1)
        detector.findHands(img, draw=False)
        lm_list = detector.findPosition(img, draw=False)
        hand_ok = bool(lm_list)

        if lm_list:
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x2 - x1, y2 - y1)

            cv2.line(img, (x1, y1), (x2, y2), CYAN, 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), 12, ORANGE, cv2.FILLED)

            vol_pct = smooth_value(
                vol_history,
                np.interp(length, [PINCH_MIN, PINCH_MAX], [0, 100]),
            )
            level = percent_to_db(vol_pct, min_vol, max_vol)
            volume.SetMasterVolumeLevel(level, None)

            if length < PINCH_MIN:
                cv2.circle(img, (cx, cy), 16, (0, 255, 255), 2)

        canvas = build_layout(img)

        now = time.time()
        fps = 1 / max(now - prev_time, 0.001)
        prev_time = now

        draw_top_bar(canvas, fps)
        draw_volume_bar(canvas, vol_pct)
        draw_status(canvas, hand_ok, music_status)

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_demo_music()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
