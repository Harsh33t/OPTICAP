#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# install.sh — OPTICap Automated Installation Script
# Team Overclocked Minds | B.Tech CSE 2026
# Run as: bash install.sh
# Requires: Raspberry Pi OS Lite 64-bit, internet connection for first run
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

OPTICAP_HOME="/home/opticap"
MODELS_DIR="$OPTICAP_HOME/models"
LOG_DIR="/var/log/opticap"

echo "═══════════════════════════════════════════"
echo "  OPTICap Installation Script"
echo "  Team Overclocked Minds | B.Tech CSE 2026"
echo "═══════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────
echo "[1/8] Installing system packages…"
sudo apt update -y
sudo apt install -y \
    python3-pip \
    python3-opencv \
    libgpiod2 \
    tesseract-ocr \
    portaudio19-dev \
    ffmpeg \
    libatlas-base-dev \
    libjpeg-dev \
    libtiff-dev \
    libopenjp2-7 \
    libavformat-dev \
    libswscale-dev \
    libcamera-apps \
    espeak \
    aplay \
    i2c-tools \
    git \
    curl \
    wget

# ── 2. Python packages ────────────────────────────────────────────
echo "[2/8] Installing Python packages…"
pip3 install --upgrade pip

pip3 install \
    tflite-runtime \
    opencv-python-headless \
    mediapipe \
    face_recognition \
    piper-tts \
    pyaudio \
    RPi.GPIO \
    numpy \
    scipy \
    Pillow \
    pyzbar \
    scikit-learn \
    pytesseract \
    openai \
    smbus2

# Optional: ultralytics (only for model export, NOT for inference on Pi)
# pip3 install ultralytics   ← run this on a dev machine, not Pi

# ── 3. Create directory structure ────────────────────────────────
echo "[3/8] Creating directory structure…"
sudo mkdir -p "$MODELS_DIR"
sudo mkdir -p "$OPTICAP_HOME/known_faces"
sudo mkdir -p "$LOG_DIR"
sudo chown -R pi:pi "$OPTICAP_HOME"
sudo chown -R pi:pi "$LOG_DIR"

# ── 4. Download AI models ─────────────────────────────────────────
echo "[4/8] Downloading AI models…"

# YAMNet audio classifier (Google, public)
YAMNET_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/yamnet_audio_classification.tflite"
if [ ! -f "$MODELS_DIR/yamnet.tflite" ]; then
    echo "  Downloading YAMNet…"
    wget -q --show-progress -O "$MODELS_DIR/yamnet.tflite" "$YAMNET_URL" || \
        echo "  WARNING: YAMNet download failed. Download manually."
else
    echo "  YAMNet already present."
fi

# YOLOv8-Nano INT8 TFLite (export guide in README.md)
if [ ! -f "$MODELS_DIR/yolov8n_int8.tflite" ]; then
    echo "  YOLOv8-Nano TFLite model not found."
    echo "  Export from a dev machine (see README.md Section 3) and copy to:"
    echo "    $MODELS_DIR/yolov8n_int8.tflite"
fi

# ── 5. Piper TTS voice ────────────────────────────────────────────
echo "[5/8] Setting up Piper TTS voice…"
PIPER_VOICE_DIR="$OPTICAP_HOME/models/piper"
mkdir -p "$PIPER_VOICE_DIR"

# Try Indian English voice, fallback to en_US
PIPER_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_JSON_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

if [ ! -f "$PIPER_VOICE_DIR/en_US-lessac-medium.onnx" ]; then
    echo "  Downloading Piper voice (en_US-lessac-medium)…"
    wget -q --show-progress -O "$PIPER_VOICE_DIR/en_US-lessac-medium.onnx" "$PIPER_VOICE_URL" || \
        echo "  WARNING: Voice download failed. Download manually from https://github.com/rhasspy/piper/blob/master/VOICES.md"
    wget -q -O "$PIPER_VOICE_DIR/en_US-lessac-medium.onnx.json" "$PIPER_JSON_URL" || true
else
    echo "  Piper voice already present."
fi

# ── 6. Copy OPTICap source files ──────────────────────────────────
echo "[6/8] Copying OPTICap source to $OPTICAP_HOME…"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo cp -r "$SCRIPT_DIR/"*.py "$OPTICAP_HOME/"
sudo chown -R pi:pi "$OPTICAP_HOME"

# ── 7. Enable required interfaces ─────────────────────────────────
echo "[7/8] Enabling camera, I2C, SPI interfaces…"
sudo raspi-config nonint do_camera 0   2>/dev/null || true
sudo raspi-config nonint do_i2c 0      2>/dev/null || true
sudo raspi-config nonint do_spi 0      2>/dev/null || true

# Add pi to gpio group
sudo usermod -aG gpio pi 2>/dev/null || true
sudo usermod -aG video pi 2>/dev/null || true
sudo usermod -aG audio pi 2>/dev/null || true

# ── 8. Systemd service ────────────────────────────────────────────
echo "[8/8] Installing systemd service…"
sudo cp "$SCRIPT_DIR/opticap.service" /etc/systemd/system/opticap.service
sudo systemctl daemon-reload
sudo systemctl enable opticap.service

echo ""
echo "═══════════════════════════════════════════"
echo "  Installation Complete!"
echo ""
echo "  NEXT STEPS:"
echo "  1. Copy YOLOv8 model: $MODELS_DIR/yolov8n_int8.tflite"
echo "  2. Register known faces: python3 $OPTICAP_HOME/register_face.py"
echo "  3. Start manually:  sudo systemctl start opticap"
echo "  4. View logs:       journalctl -u opticap -f"
echo "  5. Reboot to auto-start: sudo reboot"
echo "═══════════════════════════════════════════"
