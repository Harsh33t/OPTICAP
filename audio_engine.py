"""
audio_engine.py — OPTICap Sound Classification & Haptic Mapping (Deaf Mode)
Team Overclocked Minds | B.Tech CSE 2026

Captures audio via USB mic → YAMNet TFLite → haptic feedback.
5-class classifier: siren, car_horn, fire_alarm, dog_bark, human_shout.
"""

import time
import threading
import logging
import numpy as np
from typing import Optional
import config
from alert_queue import alert_manager
from haptic_engine import haptic_engine

logger = logging.getLogger("opticap.audio")

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pyaudio
    _PYAUDIO_OK = True
except ImportError:
    _PYAUDIO_OK = False
    logger.warning("pyaudio not installed — audio engine disabled.")

try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_OK = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        _TFLITE_OK = True
    except ImportError:
        _TFLITE_OK = False
        logger.error("tflite_runtime not available — YAMNet disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# YAMNet class index → friendly label mapping (subset of 521 classes)
# These indices correspond to AudioSet / YAMNet class ontology.
# ─────────────────────────────────────────────────────────────────────────────
YAMNET_TARGET_CLASSES = {
    # (index, label, haptic_key)
    # Indices approximate — verify against yamnet_class_map.csv
    "siren":       (396, "siren"),
    "car_horn":    (300, "car_horn"),
    "fire_alarm":  (401, "fire_alarm"),
    "dog_bark":    (74,  "dog_bark"),
    "human_shout": (0,   "human_shout"),    # Index 0 = "Speech" as fallback
}

# Build fast lookup: yamnet_idx → key
_IDX_TO_KEY: dict[int, str] = {v[0]: k for k, v in YAMNET_TARGET_CLASSES.items()}


class AudioEngine:
    """
    Polls USB microphone in 0.5-second windows.
    Classifies with YAMNet TFLite.
    Maps sound class to directional haptic pattern.
    """

    CHUNK_SAMPLES = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_SEC)

    def __init__(self):
        self._interp     = None
        self._input_idx  = None
        self._output_idx = None
        self._pa: Optional["pyaudio.PyAudio"] = None
        self._stream     = None
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._cooldowns: dict[str, float] = {}     # class → last trigger time

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def load_model(self) -> bool:
        import os
        if not _TFLITE_OK:
            return False
        if not os.path.exists(config.SOUND_MODEL_PATH):
            logger.error(f"YAMNet model not found at {config.SOUND_MODEL_PATH}")
            return False
        try:
            self._interp = tflite.Interpreter(
                model_path=config.SOUND_MODEL_PATH, num_threads=2)
            self._interp.allocate_tensors()
            self._input_idx  = self._interp.get_input_details()[0]["index"]
            self._output_idx = self._interp.get_output_details()[0]["index"]
            logger.info("YAMNet TFLite model loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load YAMNet: {e}")
            return False

    def open_microphone(self) -> bool:
        if not _PYAUDIO_OK:
            return False
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=config.AUDIO_CHANNELS,
                rate=config.AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SAMPLES,
            )
            logger.info("Microphone opened.")
            return True
        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            return False

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="AudioEngine", daemon=True)
        self._thread.start()
        logger.info("AudioEngine started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("AudioEngine stopped.")

    # ── Main Loop ─────────────────────────────────────────────────────────
    def _loop(self):
        while self._running:
            if self._stream is None:
                time.sleep(0.5)
                continue

            try:
                raw = self._stream.read(self.CHUNK_SAMPLES, exception_on_overflow=False)
                waveform = np.frombuffer(raw, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Audio read error: {e}")
                time.sleep(0.1)
                continue

            if self._interp is None:
                time.sleep(0.1)
                continue

            label, confidence = self._classify(waveform)
            if label and confidence >= config.AUDIO_CONFIDENCE:
                self._handle_detection(label, confidence)

    # ── YAMNet Inference ──────────────────────────────────────────────────
    def _classify(self, waveform: np.ndarray) -> tuple[Optional[str], float]:
        """
        Returns (label, confidence) for the highest-scoring target class.
        YAMNet expects mono float32 waveform at 16kHz.
        """
        try:
            # YAMNet input shape: [waveform_length] — flatten to 1D
            waveform = waveform.astype(np.float32)
            # Resize if model expects fixed input
            inp_details = self._interp.get_input_details()[0]
            expected_len = inp_details["shape"][0] if inp_details["shape"].ndim == 1 else None
            if expected_len and len(waveform) != expected_len:
                waveform = np.resize(waveform, expected_len)

            self._interp.set_tensor(self._input_idx, waveform)
            self._interp.invoke()

            scores = self._interp.get_tensor(self._output_idx)
            # scores shape: [num_frames, 521] or [521]
            if scores.ndim > 1:
                scores = scores.mean(axis=0)
            scores = scores.flatten()

            # Find best target class
            best_label, best_score = None, 0.0
            for idx, key in _IDX_TO_KEY.items():
                if idx < len(scores) and scores[idx] > best_score:
                    best_score = scores[idx]
                    best_label = key

            return best_label, float(best_score)
        except Exception as e:
            logger.error(f"YAMNet inference error: {e}")
            return None, 0.0

    # ── Detection Handling ────────────────────────────────────────────────
    def _handle_detection(self, label: str, confidence: float):
        # Cooldown check
        now = time.time()
        last = self._cooldowns.get(label, 0.0)
        if now - last < config.AUDIO_ALERT_COOLDOWN:
            return
        self._cooldowns[label] = now

        pattern_info = config.HAPTIC_PATTERNS.get(label)
        if not pattern_info:
            return

        logger.info(f"Sound detected: {label} ({confidence:.2f})")

        # Determine side (mono mic = both; stereo = amplitude diff)
        side = self._estimate_side()

        # Trigger haptic
        haptic_engine.play(pattern_info, side=side)

        # For fire alarm: also speak alert
        if label == "fire_alarm":
            alert_manager.speak(
                "Fire alarm detected, evacuate immediately",
                priority=config.PRIORITY_CRITICAL, source="audio")
        elif label == "siren":
            alert_manager.speak(
                "Emergency vehicle nearby, stay clear",
                priority=config.PRIORITY_HIGH, source="audio")

    def _estimate_side(self) -> str:
        """Mono mic → cannot estimate direction. Default to both."""
        return "both"

    # ── Simulated Input (for testing) ─────────────────────────────────────
    def inject_audio(self, waveform: np.ndarray):
        """For testing: bypass mic and inject waveform directly."""
        if self._interp is None:
            return
        label, confidence = self._classify(waveform)
        if label and confidence >= config.AUDIO_CONFIDENCE:
            self._handle_detection(label, confidence)


# ── Singleton ─────────────────────────────────────────────────────────────────
audio_engine = AudioEngine()
