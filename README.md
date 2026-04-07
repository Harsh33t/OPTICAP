# OPTICap — Integrated Edge-AI Sensory Intelligence System

**Team Overclocked Minds | B.Tech CSE 2026**

> *Seeing for the Blind, Hearing for the Deaf, Speaking for the Mute.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Software Architecture](#3-software-architecture)
4. [Installation](#4-installation)
5. [Model Export Guide](#5-model-export-guide)
6. [Calibration](#6-calibration)
7. [Face Registration](#7-face-registration)
8. [Operating Modes & Button Map](#8-operating-modes--button-map)
9. [ISL Gesture Vocabulary](#9-isl-gesture-vocabulary)
10. [Configuration Reference](#10-configuration-reference)
11. [Running & Logs](#11-running--logs)
12. [Testing](#12-testing)
13. [Memory Budget](#13-memory-budget)
14. [Performance Targets](#14-performance-targets)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. System Overview

OPTICap is a solar-assisted wearable smart cap that provides real-time sensory assistance to three underserved communities:

| User | Problem | OPTICap Solution |
|---|---|---|
| **Blind** | Cannot perceive environment | Camera + AI narrates world via bone-conduction audio |
| **Deaf** | Cannot hear hazard sounds | Ambient sounds mapped to directional haptic vibration |
| **Mute** | Cannot speak | ISL hand gestures converted to spoken words |

Everything runs **entirely offline** on a Raspberry Pi 4. No cloud. No GUI. CLI + hardware I/O only.

---

## 2. Hardware Requirements

| Component | Spec |
|---|---|
| SBC | Raspberry Pi 4 (2GB RAM), 64-bit Lite OS |
| Camera | Pi Camera Module v2 (5MP, 160° wide-angle, ~15° downward tilt) |
| Ultrasonic | 2× REES52 HC-SR04 (left + right) |
| Microphone | USB microphone |
| Audio Out | Bone-conduction earbud (3.5mm jack) |
| Haptic | 2× 10mm coin vibration motors (GPIO-driven, PWM) |
| Buzzer | 5V Active Piezo Buzzer (85dB) |
| Buttons | 3× fabric push buttons (GPIO, active LOW with PUD_UP) |
| Power | 2× 3.7V 2500mAh Li-ion (5000mAh total), TP4056 charger, 5V/3A boost |

### GPIO Pin Map (BCM numbering)

```
ULTRASONIC LEFT  : TRIG=11, ECHO=13
ULTRASONIC RIGHT : TRIG=15, ECHO=16
BUTTON MODE      : GPIO 17
BUTTON SOS       : GPIO 27
BUTTON GESTURE   : GPIO 22
MOTOR LEFT       : GPIO 23  (PWM)
MOTOR RIGHT      : GPIO 24  (PWM)
BUZZER           : GPIO 25
```

---

## 3. Software Architecture

```
main_controller.py           ← Master orchestrator
├── vision_engine.py         ← Camera + YOLOv8-Nano + OCR trigger
│   └── pothole_detector.py  ← Specialized road hazard detection
├── audio_engine.py          ← USB Mic + YAMNet sound classification
├── gesture_engine.py        ← MediaPipe Hands + ISL-to-Speech
├── face_recognition_engine.py ← Face detection + recognition
├── haptic_engine.py         ← Vibration motors + Buzzer controller
├── tts_engine.py            ← Piper TTS (offline, priority queue)
├── ultrasonic_engine.py     ← Dual HC-SR04 blind-spot coverage
├── button_handler.py        ← GPIO buttons + mode controller
├── alert_queue.py           ← Priority alert manager
├── depth_estimator.py       ← Monocular depth + step count
├── config.py                ← All constants, GPIO pins, flags
├── mock_gpio.py             ← MockGPIO for testing without hardware
└── register_face.py         ← CLI face registration tool
```

---

## 4. Installation

### Prerequisites

- Raspberry Pi 4 (2GB minimum) running Raspberry Pi OS Lite 64-bit
- Internet connection (first install only)
- Pi Camera Module connected and enabled

### Steps

```bash
# 1. Clone / copy source to Pi
scp -r "OPTICAP code/" pi@raspberrypi.local:/home/pi/opticap-src/

# 2. Run installer (on Pi)
cd /home/pi/opticap-src
bash install.sh
```

The installer will:
- Install all system packages (OpenCV, Tesseract, PortAudio, etc.)
- Install all Python packages
- Download YAMNet and Piper TTS voice
- Create `/home/opticap/` directory structure
- Register systemd service (`opticap.service`)
- Enable camera, I2C, SPI interfaces

---

## 5. Model Export Guide

### YOLOv8-Nano INT8 TFLite (run on dev machine, NOT on Pi)

```bash
# On a machine with GPU / full RAM
pip install ultralytics

python3 - <<'EOF'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Export to TFLite INT8 (quantized)
model.export(
    format="tflite",
    imgsz=320,
    int8=True,        # INT8 quantization → smaller, faster on Pi
    data="coco128.yaml",  # calibration dataset
)
# Output: yolov8n_int8.tflite
EOF

# Copy to Pi
scp yolov8n_int8.tflite pi@raspberrypi.local:/home/opticap/models/
```

### Pothole Detector TFLite

Train using Pothole-600 dataset or IDD (India Driving Dataset):
```bash
# After training with your chosen framework, export:
model.export(format="tflite", int8=True)
scp pothole_detector.tflite pi@raspberrypi.local:/home/opticap/models/
```

### ISL Gesture Classifier

Train an SVM/MLP on MediaPipe landmark data:
```bash
# Collect training data using /home/opticap/gesture_engine.py in data-collection mode
# Then train:
python3 train_gesture.py   # produces isl_gesture.pkl
scp isl_gesture.pkl pi@raspberrypi.local:/home/opticap/models/
```

---

## 6. Calibration

### Camera Focal Length (for face distance)

1. Have a person stand exactly **1 metre** from the camera.
2. Run:
   ```bash
   python3 - <<'EOF'
   import cv2, face_recognition
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   cap.release()
   rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   locations = face_recognition.face_locations(rgb)
   if locations:
       top, right, bottom, left = locations[0]
       pixel_width = right - left
       known_width_cm = 14.0   # average face width
       focal_length = pixel_width  # at 1m, focal_length = pixel_width / 1
       print(f"FOCAL_LENGTH_PX = {focal_length}")
   EOF
   ```
3. Update `FOCAL_LENGTH_PX` in `config.py`.

### Camera Height

Measure actual camera mounting height from ground. Update `CAMERA_HEIGHT_M` in `config.py`.

---

## 7. Face Registration

Run this tool to add a known person to the recognition database:

```bash
python3 /home/opticap/register_face.py
```

- Prompts for the person's name
- Captures 5 photos automatically (1 per second)
- Saves to `/home/opticap/known_faces/<name>/`
- Rebuilds the encodings database at `/home/opticap/known_faces/encodings.pkl`

**Disable face recognition** (privacy mode) by setting in `config.py`:
```python
FACE_RECOGNITION_ENABLED = False
```

---

## 8. Operating Modes & Button Map

### Modes

| Mode | Active Modules |
|---|---|
| **VISION** | Camera + Ultrasonic + Face |
| **DEAF** | Audio classification + Haptic |
| **MUTE** | Gesture recognition + TTS |
| **ALL** | All modules simultaneously |

### Button Actions

| Button | GPIO | Short Press | Long Press (>2s) |
|---|---|---|---|
| **Mode** | 17 | Cycle modes: VISION → DEAF → MUTE → ALL | Mute/unmute all alerts |
| **SOS** | 27 | Read text in view (OCR) | Trigger SOS alert sequence |
| **Gesture** | 22 | Toggle gesture recognition | Read battery level |

---

## 9. ISL Gesture Vocabulary

| Gesture | Spoken Output |
|---|---|
| Open palm | "Hello" |
| Thumbs up | "Yes. Okay." |
| Thumbs down | "No" |
| Pointing index finger | "Help me" |
| Fist | "Stop" |
| Peace sign (V) | "Thank you" |
| Pinky extended | "Water" |
| Open hand left-right wave | "My name is [User]" |
| Index + middle crossed | "I need a doctor" |
| Open hand moving left-right | "I don't understand" |

Gesture must be held stable for **0.8 seconds** before triggering (prevents accidental activation).

---

## 10. Configuration Reference

All settings are in `config.py`. Key parameters:

```python
# Identity
USER_NAME = "User"          # Used in gesture "My name is..."

# Startup mode
MODE = "ALL"                # VISION | DEAF | MUTE | ALL

# Confidence thresholds
YOLO_CONFIDENCE       = 0.50   # daytime detection threshold
YOLO_CONFIDENCE_NIGHT = 0.35   # night mode (lower = more sensitive)
FACE_MATCH_TOLERANCE  = 0.50   # face recognition (lower = stricter)
AUDIO_CONFIDENCE      = 0.65   # sound classification threshold

# Feature flags
FACE_RECOGNITION_ENABLED   = True
POTHOLE_LOGGING_ENABLED    = False   # requires GPS module
AGE_GENDER_ENABLED         = False
FALL_DETECTION_ENABLED     = False   # requires MPU6050 IMU
```

---

## 11. Running & Logs

```bash
# Start manually
sudo systemctl start opticap

# Stop
sudo systemctl stop opticap

# View live logs
journalctl -u opticap -f

# Alert log (structured, all alerts with timestamps)
tail -f /var/log/opticap/alerts.log

# System log
tail -f /var/log/opticap/system.log

# Run directly (for debugging)
cd /home/opticap
python3 main_controller.py
```

---

## 12. Testing

```bash
# Run full test suite (no hardware required — uses MockGPIO)
cd /home/opticap
python3 test_suite.py

# Run 5-minute memory stress test (on Pi hardware only)
python3 test_suite.py --stress 300
```

### Test Coverage

| Test Class | What It Tests |
|---|---|
| `TestConfig` | GPIO pin uniqueness, priority ordering |
| `TestDepthEstimator` | Monocular depth formula, direction logic |
| `TestAlertQueue` | Priority queuing, CRITICAL bypass, LOW drop |
| `TestPotholeDetector` | Severity classification, night mode, water detection |
| `TestVisionEngine` | Tier lookup, traffic light color, deduplication |
| `TestGestureEngine` | Rule-based classifier, feature extraction |
| `TestMockGPIO` | Full GPIO mock API coverage |
| `TestAlertDeduplication` | Same-object repeated detection → single announce |
| `TestLatency` | Detection → TTS dispatch < 500ms |
| `TestMemoryStress` | Sustained load, no deadlocks |

---

## 13. Memory Budget

| Component | Allocated |
|---|---|
| OS Lite (headless) | ~180 MB |
| OpenCV (320×320 feed) | ~90 MB |
| YOLOv8-Nano INT8 | ~55 MB |
| Tesseract OCR (on-demand) | ~80 MB |
| Piper TTS (single voice) | ~40 MB |
| GPIO + threads + misc | ~120 MB |
| **Total** | **~565 MB** |

System enforces `MemoryMax=600M` in the systemd service unit.

---

## 14. Performance Targets

| Metric | Target | How Achieved |
|---|---|---|
| Detection → audio | < 500ms | Priority queue + daemon TTS threads |
| Frame processing rate | 10–15 FPS | 320×320 input, INT8 quantized model |
| Peak RAM | < 600MB | Lazy imports (gesture, OCR), INT8 model |
| TTS startup | < 200ms | Piper raw PCM piped to aplay (no file I/O) |
| Ultrasonic polling | 200ms | Separate sensor threads |
| Gesture recognition | < 1 second | 0.8s hold check in tight loop |
| Face recognition | Every 5th frame | `FACE_DETECTION_FRAME_SKIP = 5` |
| Boot to ready | < 30 seconds | systemd, no desktop environment |

---

## 15. Troubleshooting

### Camera not opening
```bash
# Check camera is enabled
vcgencmd get_camera
# Should show: supported=1 detected=1

# Enable in raspi-config
sudo raspi-config → Interface Options → Camera → Enable
```

### GPIO permission denied
```bash
sudo usermod -aG gpio pi
# Log out and back in, or reboot
```

### Piper TTS silent
```bash
# Test audio output
aplay /usr/share/sounds/alsa/Front_Center.wav

# Check voice file exists
ls /home/opticap/models/piper/

# Test piper directly
echo "Hello world" | piper --model /home/opticap/models/piper/en_US-lessac-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -c 1 -
```

### YOLOv8 model not loading
```bash
# Check file exists and is correct size (should be ~3-6 MB)
ls -lh /home/opticap/models/yolov8n_int8.tflite

# Test with Python
python3 -c "import tflite_runtime.interpreter as tflite; i = tflite.Interpreter('/home/opticap/models/yolov8n_int8.tflite'); print('OK')"
```

### Face recognition slow
- Ensure you're running on every **5th frame** (check `FACE_DETECTION_FRAME_SKIP` in config)
- Consider switching to MediaPipe-only mode (set `FACE_RECOGNITION_ENABLED = False`)

---

*OPTICap — Seeing for the Blind, Hearing for the Deaf, Speaking for the Mute.*
*Team Overclocked Minds | B.Tech CSE 2026*
