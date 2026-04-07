"""
config.py — OPTICap Master Configuration File
Team Overclocked Minds | B.Tech CSE 2026
All constants, GPIO pins, thresholds, paths, and feature flags.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE — Camera
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0              # /dev/video0 (Pi Camera via v4l2)
CAMERA_WIDTH         = 320
CAMERA_HEIGHT        = 320
CAMERA_FPS           = 15
CAMERA_HEIGHT_M      = 1.6           # Height of camera from ground (metres)
CAMERA_TILT_DEG      = 15            # Downward tilt angle (degrees)
CAMERA_VFOV_DEG      = 41.4          # Pi Camera v2 vertical FOV
CAMERA_HFOV_DEG      = 53.5          # Pi Camera v2 horizontal FOV

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE — GPIO Pin Numbers (BCM numbering)
# ─────────────────────────────────────────────────────────────────────────────
ULTRASONIC_LEFT_TRIG  = 11
ULTRASONIC_LEFT_ECHO  = 13
ULTRASONIC_RIGHT_TRIG = 15
ULTRASONIC_RIGHT_ECHO = 16

BUTTON_MODE           = 17           # Button 1: Mode cycling
BUTTON_SOS            = 27           # Button 2: SOS / OCR read
BUTTON_GESTURE        = 22           # Button 3: Gesture toggle / Battery

MOTOR_LEFT            = 23           # Left vibration motor (PWM capable)
MOTOR_RIGHT           = 24           # Right vibration motor (PWM capable)
BUZZER_PIN            = 25           # Active piezo buzzer

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE — Face Calibration
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_FACE_WIDTH_CM   = 14.0         # Average human face width (cm)
FOCAL_LENGTH_PX       = 600.0        # Calibrated focal length in pixels
                                     # (run calibration at 1m with known face)

# ─────────────────────────────────────────────────────────────────────────────
# AI MODEL PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR              = "/home/opticap"
MODELS_DIR            = os.path.join(BASE_DIR, "models")

YOLO_MODEL_PATH       = os.path.join(MODELS_DIR, "yolov8n_int8.tflite")
POTHOLE_MODEL_PATH    = os.path.join(MODELS_DIR, "pothole_detector.tflite")
SOUND_MODEL_PATH      = os.path.join(MODELS_DIR, "yamnet.tflite")
GESTURE_MODEL_PATH    = os.path.join(MODELS_DIR, "isl_gesture.pkl")
FACE_ENCODINGS_PATH   = os.path.join(BASE_DIR, "known_faces", "encodings.pkl")
KNOWN_FACES_DIR       = os.path.join(BASE_DIR, "known_faces")

# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────
TTS_VOICE             = "en_IN-male-medium"   # Piper voice identifier
TTS_SPEED             = 1.2                   # Speak 20% faster than default

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
YOLO_CONFIDENCE            = 0.50
YOLO_CONFIDENCE_NIGHT      = 0.35
FACE_MATCH_TOLERANCE       = 0.50      # Lower = stricter match
AUDIO_CONFIDENCE           = 0.65
GESTURE_HOLD_SEC           = 0.80      # Gesture must be stable for this long

# Vision alert deduplication
DEDUP_WINDOW_SEC           = 4.0       # Same object+direction suppressed for N sec
DEDUP_HISTORY_SIZE         = 5         # Track last N announcements

# ─────────────────────────────────────────────────────────────────────────────
# ULTRASONIC ALERT THRESHOLDS (cm)
# ─────────────────────────────────────────────────────────────────────────────
ULTRASONIC_CRITICAL_CM     = 30
ULTRASONIC_MEDIUM_CM       = 80
ULTRASONIC_SOFT_CM         = 150
ULTRASONIC_POLL_INTERVAL   = 0.20      # Seconds between readings

# ─────────────────────────────────────────────────────────────────────────────
# ALERT COOLDOWNS (seconds)
# ─────────────────────────────────────────────────────────────────────────────
VEHICLE_ALERT_COOLDOWN     = 2.0
GROUND_HAZARD_COOLDOWN     = 3.0
STRUCTURAL_COOLDOWN        = 4.0
INFORMATIONAL_COOLDOWN     = 3.0
AUDIO_ALERT_COOLDOWN       = 3.0
ULTRASONIC_SUPPRESS_SEC    = 1.0       # Suppress ultrasonic after camera alert

# ─────────────────────────────────────────────────────────────────────────────
# BRIGHTNESS / NIGHT MODE
# ─────────────────────────────────────────────────────────────────────────────
NIGHT_MODE_BRIGHTNESS      = 60        # Mean pixel value below which night mode triggers

# ─────────────────────────────────────────────────────────────────────────────
# MONOCULAR DEPTH
# ─────────────────────────────────────────────────────────────────────────────
STRIDE_LENGTH_M            = 0.65      # Average walking stride (metres)
MIN_DISTANCE_M             = 0.30      # Clamp floor to avoid division artifacts

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE          = 16000
AUDIO_CHANNELS             = 1
AUDIO_CHUNK_SEC            = 0.5       # Process audio in 0.5-second windows

# ─────────────────────────────────────────────────────────────────────────────
# OPERATING MODES
# ─────────────────────────────────────────────────────────────────────────────
MODE_VISION  = "VISION"
MODE_DEAF    = "DEAF"
MODE_MUTE    = "MUTE"
MODE_ALL     = "ALL"
MODES_ORDER  = [MODE_VISION, MODE_DEAF, MODE_MUTE, MODE_ALL]

MODE         = MODE_ALL                # Default startup mode

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FLAGS
# ─────────────────────────────────────────────────────────────────────────────
FACE_RECOGNITION_ENABLED   = True
POTHOLE_LOGGING_ENABLED    = False     # Requires GPS module
AGE_GENDER_ENABLED         = False
FALL_DETECTION_ENABLED     = False     # Requires MPU6050 IMU
EMOTION_DETECTION_ENABLED  = False

# ─────────────────────────────────────────────────────────────────────────────
# USER
# ─────────────────────────────────────────────────────────────────────────────
USER_NAME = "User"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR       = "/var/log/opticap"
ALERT_LOG     = os.path.join(LOG_DIR, "alerts.log")
SYSTEM_LOG    = os.path.join(LOG_DIR, "system.log")

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY ALERT LEVELS
# ─────────────────────────────────────────────────────────────────────────────
PRIORITY_CRITICAL   = 1
PRIORITY_HIGH       = 2
PRIORITY_MEDIUM     = 3
PRIORITY_LOW        = 4

# ─────────────────────────────────────────────────────────────────────────────
# YOLO CLASS PRIORITY MAP
# tier → (priority_level, cooldown_sec, message_template)
# {direction} and {steps} are substituted at runtime.
# ─────────────────────────────────────────────────────────────────────────────
PRIORITY_TIER_1 = {
    "classes": ["car", "truck", "motorcycle", "bus", "bicycle",
                 "auto_rickshaw", "tuk_tuk", "e-scooter"],
    "priority": PRIORITY_CRITICAL,
    "cooldown": VEHICLE_ALERT_COOLDOWN,
    "message": "Vehicle approaching, stop immediately",
    "buzzer": True,
    "buzzer_pattern": "rapid_triple",
}

PRIORITY_TIER_2 = {
    "classes": ["pothole", "open_drain", "manhole", "construction_debris",
                 "road_depression", "open_manhole", "garbage_pile", "speed_bump"],
    "priority": PRIORITY_HIGH,
    "cooldown": GROUND_HAZARD_COOLDOWN,
    "message_template": "{hazard} ahead, {steps} steps, {direction}",
    "buzzer": False,
}

PRIORITY_TIER_3 = {
    "classes": ["stairs_up", "stairs_down", "ramp", "curb",
                 "electric_pole", "low_hanging_wire"],
    "priority": PRIORITY_MEDIUM,
    "cooldown": STRUCTURAL_COOLDOWN,
    "message_template": "{hazard} {direction}",
    "buzzer": True,
    "buzzer_pattern": "double",
}

PRIORITY_TIER_4 = {
    "classes": ["person", "dog", "cat", "bicycle", "traffic_light", "bench",
                 "door", "table", "chair", "cow", "goat", "street_vendor_cart",
                 "traffic_signal", "pedestrian_crossing", "wet_floor",
                 "bird", "backpack", "handbag"],
    "priority": PRIORITY_LOW,
    "cooldown": INFORMATIONAL_COOLDOWN,
    "message_template": "{object} {direction}",
    "buzzer": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# HAPTIC PATTERNS (for deaf mode)
# Motor pulse durations in milliseconds
# ─────────────────────────────────────────────────────────────────────────────
HAPTIC_PATTERNS = {
    "siren":       {"pattern": "alternating_rapid", "cycles": 3, "duration_ms": 100},
    "car_horn":    {"pattern": "single_burst",      "cycles": 1, "duration_ms": 400},
    "fire_alarm":  {"pattern": "both_continuous",   "cycles": 5, "duration_ms": 200},
    "dog_bark":    {"pattern": "single_pulse",      "cycles": 1, "duration_ms": 300},
    "human_shout": {"pattern": "long_center",       "cycles": 1, "duration_ms": 800},
}

# ─────────────────────────────────────────────────────────────────────────────
# ISL GESTURE VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
ISL_GESTURE_MAP = {
    "open_palm":           "Hello",
    "thumbs_up":           "Yes. Okay.",
    "thumbs_down":         "No",
    "pointing_index":      "Help me",
    "fist":                "Stop",
    "peace_sign":          "Thank you",
    "pinky_extended":      "Water",
    "open_wave":           f"My name is {USER_NAME}",
    "index_middle_cross":  "I need a doctor",
    "open_left_right":     "I don't understand",
}

# ─────────────────────────────────────────────────────────────────────────────
# BUZZER PATTERNS (duration in seconds)
# ─────────────────────────────────────────────────────────────────────────────
BUZZER_RAPID_TRIPLE = [(0.1, 0.1), (0.1, 0.1), (0.1, 0.0)]   # (on, off) pairs
BUZZER_DOUBLE       = [(0.2, 0.2), (0.2, 0.0)]
BUZZER_SINGLE       = [(0.5, 0.0)]
BUZZER_SOS          = [(0.1,0.1)]*3 + [(0.5,0.5)]*3 + [(0.1,0.1)]*3

# ─────────────────────────────────────────────────────────────────────────────
# FACE RECOGNITION - Run face detection every N frames
# ─────────────────────────────────────────────────────────────────────────────
FACE_DETECTION_FRAME_SKIP = 5

# ─────────────────────────────────────────────────────────────────────────────
# POTHOLE SEVERITY THRESHOLDS (fraction of frame area)
# ─────────────────────────────────────────────────────────────────────────────
POTHOLE_SMALL_AREA  = 0.05
POTHOLE_MEDIUM_AREA = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# BUTTON TIMING
# ─────────────────────────────────────────────────────────────────────────────
BUTTON_DEBOUNCE_MS      = 200
BUTTON_LONG_PRESS_SEC   = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC LIGHT COLOR HSV RANGES
# ─────────────────────────────────────────────────────────────────────────────
TL_RED_LOWER1    = (0,   120, 70)
TL_RED_UPPER1    = (10,  255, 255)
TL_RED_LOWER2    = (170, 120, 70)
TL_RED_UPPER2    = (180, 255, 255)
TL_GREEN_LOWER   = (40,  50,  50)
TL_GREEN_UPPER   = (90,  255, 255)
TL_YELLOW_LOWER  = (20,  100, 100)
TL_YELLOW_UPPER  = (35,  255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = "your_token_here"
