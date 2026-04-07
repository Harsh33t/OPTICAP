"""
face_recognition_engine.py — OPTICap Face Detection & Recognition
Team Overclocked Minds | B.Tech CSE 2026

Detection: MediaPipe Face Detection (lightweight, preferred for Pi).
Recognition: dlib 128-d face encodings (face_recognition library).
Supports: known-face database, crowd count, emotion (optional), re-ID.
"""

import os
import time
import pickle
import logging
import threading
import numpy as np
import cv2
from typing import Optional
import config
from depth_estimator import distance_to_face, describe_distance
from alert_queue import alert_manager

logger = logging.getLogger("opticap.face")

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports — graceful fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False
    logger.warning("mediapipe not installed — face detection will be skipped.")

try:
    import face_recognition as fr
    _FR_OK = True
except ImportError:
    _FR_OK = False
    logger.warning("face_recognition not installed — recognition disabled.")


class FaceRecognitionEngine:
    """
    Run face detection on every Nth frame (config.FACE_DETECTION_FRAME_SKIP).
    Optionally match against known face database.
    """

    MIN_REIDENT_GAP = 5.0   # Don't re-announce same face within N seconds

    def __init__(self):
        self._mp_detector  = None
        self._known_names:    list[str]        = []
        self._known_encs:     list[np.ndarray] = []
        self._face_ids:       dict             = {}   # enc_hash → (name, last_seen)
        self._last_seen:      dict             = {}   # name → timestamp
        self._frame_counter   = 0
        self._running         = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def setup(self):
        if _MP_OK:
            mp_face = mp.solutions.face_detection
            self._mp_detector = mp_face.FaceDetection(
                model_selection=0,         # 0 = short range, faster on Pi
                min_detection_confidence=0.5,
            )
            logger.info("MediaPipe Face Detection initialised.")

        if config.FACE_RECOGNITION_ENABLED and _FR_OK:
            self._load_encodings()

    def start(self, get_frame_fn):
        """Start the face engine. `get_frame_fn` is called to get latest frame."""
        self._get_frame = get_frame_fn
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="FaceEngine", daemon=True)
        self._thread.start()
        logger.info("FaceRecognitionEngine started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── Main Loop ─────────────────────────────────────────────────────────
    def _loop(self):
        while self._running:
            self._frame_counter += 1
            if self._frame_counter % config.FACE_DETECTION_FRAME_SKIP != 0:
                time.sleep(0.05)
                continue

            frame = self._get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            faces = self._detect(frame)
            if not faces:
                time.sleep(0.05)
                continue

            # Sort by bounding box area (largest = closest)
            faces.sort(key=lambda f: f.get("area", 0), reverse=True)

            # Crowd count
            if len(faces) > 5:
                alert_manager.speak("Crowded area ahead", priority=config.PRIORITY_MEDIUM, source="face")
            elif len(faces) > 1:
                alert_manager.speak(f"{len(faces)} people nearby", priority=config.PRIORITY_LOW, source="face")

            # Process closest face first
            face = faces[0]
            self._announce_face(face, frame)

    # ── Detection ─────────────────────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> list[dict]:
        """Return list of face dicts: {bbox, area, pixel_width, encoding}"""
        results = []
        if self._mp_detector is None:
            return results

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mp_results = self._mp_detector.process(rgb)

        if not mp_results.detections:
            return results

        for det in mp_results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = int(bb.xmin * w);  y1 = int(bb.ymin * h)
            x2 = int((bb.xmin + bb.width) * w)
            y2 = int((bb.ymin + bb.height) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            pixel_width = x2 - x1

            enc = None
            if config.FACE_RECOGNITION_ENABLED and _FR_OK:
                # Re-encode using face_recognition (dlib)
                face_roi = rgb[y1:y2, x1:x2]
                encs = fr.face_encodings(rgb, [(y1, x2, y2, x1)])
                enc = encs[0] if encs else None

            results.append({
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "pixel_width": pixel_width,
                "encoding": enc,
            })
        return results

    # ── Recognition & Announcement ────────────────────────────────────────
    def _announce_face(self, face: dict, frame: np.ndarray):
        pixel_width = face["pixel_width"]
        dist_cm     = distance_to_face(pixel_width)
        dist_m      = dist_cm / 100.0
        dist_str    = describe_distance(dist_m)

        name = "Unknown person"
        if (config.FACE_RECOGNITION_ENABLED and _FR_OK
                and face["encoding"] is not None and self._known_encs):
            matches = fr.compare_faces(
                self._known_encs, face["encoding"],
                tolerance=config.FACE_MATCH_TOLERANCE)
            if any(matches):
                idx  = np.argmin(fr.face_distance(self._known_encs, face["encoding"]))
                name = self._known_names[idx]

        # Suppress repeat announcements
        last = self._last_seen.get(name, 0.0)
        if time.time() - last < self.MIN_REIDENT_GAP:
            return
        self._last_seen[name] = time.time()

        if name == "Unknown person":
            msg = f"Unknown person, {dist_str}"
        else:
            msg = f"{name} is nearby, {dist_str}"
        alert_manager.speak(msg, priority=config.PRIORITY_LOW, source="face")

    # ── Face Database ─────────────────────────────────────────────────────
    def _load_encodings(self):
        if not os.path.exists(config.FACE_ENCODINGS_PATH):
            logger.info("No face encodings file found. Run register_face.py to add faces.")
            return
        try:
            with open(config.FACE_ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            self._known_names = data["names"]
            self._known_encs  = data["encodings"]
            logger.info(f"Loaded {len(self._known_names)} known faces.")
        except Exception as e:
            logger.error(f"Failed to load face encodings: {e}")

    def reload_encodings(self):
        """Hot-reload encodings without restart (called after registration)."""
        self._load_encodings()


# ── Singleton ─────────────────────────────────────────────────────────────────
face_engine = FaceRecognitionEngine()
