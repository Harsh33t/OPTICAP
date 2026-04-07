"""
gesture_engine.py — OPTICap ISL Gesture Recognition Engine (Mute Mode)
Team Overclocked Minds | B.Tech CSE 2026

MediaPipe Hands → 63 landmark features → SVM/MLP classifier → TTS output.
Activates only when Button 3 is pressed (saves ~80MB RAM when idle).
"""

import os
import time
import pickle
import threading
import logging
import numpy as np
import subprocess
from typing import Optional
import config
from alert_queue import alert_manager

logger = logging.getLogger("opticap.gesture")

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False
    logger.warning("mediapipe not installed — gesture engine disabled.")

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    from sklearn.svm import SVC
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    logger.warning("scikit-learn not installed — gesture classifier disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Hand landmark indices (MediaPipe standard)
# ─────────────────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 21
FEATURE_DIM   = NUM_LANDMARKS * 3   # x, y, z per landmark


class GestureEngine:
    """
    ISL gesture recognition using MediaPipe + sklearn classifier.
    Must be explicitly started/stopped via `activate()` / `deactivate()`.
    """

    def __init__(self):
        self._hands       = None
        self._classifier  = None
        self._label_names: list[str] = []
        self._active      = False
        self._thread: Optional[threading.Thread] = None
        self._cap         = None
        self._last_gesture: Optional[str] = None
        self._gesture_start_time: float = 0.0
        self._last_triggered: Optional[str] = None
        self._last_trigger_time: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def load_model(self) -> bool:
        """Load the ISL gesture classifier (sklearn pkl)."""
        if not os.path.exists(config.GESTURE_MODEL_PATH):
            logger.warning(
                f"Gesture model not found at {config.GESTURE_MODEL_PATH}. "
                "Will use rule-based fallback.")
            return False
        if not _SKLEARN_OK:
            logger.error("scikit-learn required for gesture classifier.")
            return False
        try:
            with open(config.GESTURE_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self._classifier  = data["model"]
            self._label_names = data["labels"]
            logger.info(f"Gesture classifier loaded. Classes: {self._label_names}")
            return True
        except Exception as e:
            logger.error(f"Failed to load gesture model: {e}")
            return False

    def activate(self, camera_index: int = 0):
        """Start gesture recognition. Called when Button 3 pressed."""
        if self._active:
            return
        if not _MP_OK or not _CV2_OK:
            alert_manager.speak("Gesture mode not available", source="gesture",
                                priority=config.PRIORITY_LOW)
            return

        import cv2
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self._active = True
        self._thread = threading.Thread(
            target=self._loop, name="GestureEngine", daemon=True)
        self._thread.start()
        self._play_ding()
        alert_manager.speak("Gesture mode active", priority=config.PRIORITY_LOW,
                            source="gesture")
        logger.info("GestureEngine activated.")

    def deactivate(self):
        """Stop gesture recognition. Frees ~80MB RAM."""
        if not self._active:
            return
        self._active = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._hands:
            self._hands.close()
            self._hands = None
        self._play_ding()
        alert_manager.speak("Gesture mode off", priority=config.PRIORITY_LOW,
                            source="gesture")
        logger.info("GestureEngine deactivated.")

    def is_active(self) -> bool:
        return self._active

    # ── Main Loop ─────────────────────────────────────────────────────────
    def _loop(self):
        import cv2
        while self._active:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)

            if not results.multi_hand_landmarks:
                self._last_gesture = None
                self._gesture_start_time = 0.0
                continue

            landmarks = results.multi_hand_landmarks[0]
            features  = self._extract_features(landmarks)
            gesture   = self._classify(features)

            if gesture is None:
                self._last_gesture = None
                continue

            # Hold-time check: gesture must be stable for GESTURE_HOLD_SEC
            now = time.time()
            if gesture == self._last_gesture:
                held = now - self._gesture_start_time
                if held >= config.GESTURE_HOLD_SEC:
                    self._trigger(gesture)
            else:
                self._last_gesture = gesture
                self._gesture_start_time = now

    # ── Feature Extraction ────────────────────────────────────────────────
    @staticmethod
    def _extract_features(landmarks) -> np.ndarray:
        """
        Flatten 21 landmarks (x, y, z) into a 63-d vector.
        Normalise relative to wrist (landmark 0).
        """
        coords = []
        wrist  = landmarks.landmark[0]
        for lm in landmarks.landmark:
            coords.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z,
            ])
        return np.array(coords, dtype=np.float32)

    # ── Classification ────────────────────────────────────────────────────
    def _classify(self, features: np.ndarray) -> Optional[str]:
        if self._classifier is not None:
            try:
                pred = self._classifier.predict([features])[0]
                return pred
            except Exception as e:
                logger.error(f"Gesture classifier error: {e}")
                return None

        # ── Rule-based fallback ────────────────────────────────────────
        return self._rule_based(features)

    @staticmethod
    def _rule_based(features: np.ndarray) -> Optional[str]:
        """
        Simple heuristic classifier based on finger curl states.
        Works without a trained model.
        """
        # Reshape to [21, 3]
        pts = features.reshape(NUM_LANDMARKS, 3)

        def finger_extended(tip_idx: int, pip_idx: int) -> bool:
            return pts[tip_idx][1] < pts[pip_idx][1]  # tip above pip (Y inverted)

        thumb_up   = pts[4][0] > pts[3][0]     # thumb tip right of thumb IP
        index_ext  = finger_extended(8,  6)
        middle_ext = finger_extended(12, 10)
        ring_ext   = finger_extended(16, 14)
        pinky_ext  = finger_extended(20, 18)

        extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        if extended == 4:
            return "open_palm"
        if extended == 0 and not thumb_up:
            return "fist"
        if thumb_up and extended == 0:
            return "thumbs_up"
        if not thumb_up and extended == 0:
            return "thumbs_down"
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "pointing_index"
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "peace_sign"
        if not index_ext and not middle_ext and not ring_ext and pinky_ext:
            return "pinky_extended"
        return None

    # ── Trigger ───────────────────────────────────────────────────────────
    def _trigger(self, gesture: str):
        now = time.time()
        if (gesture == self._last_triggered and
                now - self._last_trigger_time < 3.0):
            return   # Avoid repeating same gesture immediately
        self._last_triggered  = gesture
        self._last_trigger_time = now

        phrase = config.ISL_GESTURE_MAP.get(gesture)
        if phrase:
            # Update dynamic phrase (user name may have changed)
            if gesture == "open_wave":
                phrase = f"My name is {config.USER_NAME}"
            alert_manager.speak(phrase, priority=config.PRIORITY_HIGH, source="gesture")
            logger.info(f"Gesture triggered: {gesture} → {phrase!r}")
        else:
            logger.debug(f"Unknown gesture: {gesture}")

        # Reset hold state
        self._last_gesture    = None
        self._gesture_start_time = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _play_ding():
        """Play a short audio cue using aplay."""
        try:
            subprocess.Popen(
                ["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass   # Ding is cosmetic; failure is acceptable


# ── Singleton ─────────────────────────────────────────────────────────────────
gesture_engine = GestureEngine()
