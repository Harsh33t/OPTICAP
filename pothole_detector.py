"""
pothole_detector.py — OPTICap Specialized Road Hazard Detection Module
Team Overclocked Minds | B.Tech CSE 2026

Primary: custom TFLite pothole model.
Fallback: YOLOv8-Nano class filtering + OpenCV texture analysis.
Includes night-mode boost, severity classification, and directional guidance.
"""

import cv2
import numpy as np
import time
import logging
import math
from typing import Optional
import config
from depth_estimator import estimate_distance, describe_distance

logger = logging.getLogger("opticap.pothole")

# ─────────────────────────────────────────────────────────────────────────────
# TFLite Runtime import
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_OK = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        _TFLITE_OK = True
    except ImportError:
        _TFLITE_OK = False
        logger.warning("tflite_runtime not available — pothole model disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# Class labels for custom pothole model
# ─────────────────────────────────────────────────────────────────────────────
POTHOLE_CLASSES = [
    "pothole", "open_drain", "road_depression",
    "construction_debris", "manhole", "open_manhole",
]


class PotholeDetector:
    """
    Detects potholes, drains, and road hazards.
    Can be called from vision_engine for every frame or on demand.
    """

    def __init__(self):
        self._interp   = None
        self._input_idx  = None
        self._output_boxes = None
        self._output_classes = None
        self._output_scores = None
        self._input_shape = (320, 320)
        self._night_announced = False
        self._last_detect_time = 0.0
        self._cooldown = config.GROUND_HAZARD_COOLDOWN
        self._model_available = False

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def load_model(self) -> bool:
        """Load the TFLite pothole model. Returns True if successful."""
        if not _TFLITE_OK:
            return False
        import os
        if not os.path.exists(config.POTHOLE_MODEL_PATH):
            logger.warning(
                f"Pothole model not found at {config.POTHOLE_MODEL_PATH}. "
                "Using OpenCV fallback.")
            return False
        try:
            self._interp = tflite.Interpreter(
                model_path=config.POTHOLE_MODEL_PATH,
                num_threads=2,
            )
            self._interp.allocate_tensors()
            input_details  = self._interp.get_input_details()
            output_details = self._interp.get_output_details()
            self._input_idx = input_details[0]["index"]
            # Typical SSD output order: boxes, classes, scores, count
            self._output_boxes   = output_details[0]["index"]
            self._output_classes = output_details[1]["index"]
            self._output_scores  = output_details[2]["index"]
            h, w = input_details[0]["shape"][1:3]
            self._input_shape = (w, h)
            self._model_available = True
            logger.info("Pothole TFLite model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load pothole model: {e}")
            return False

    # ── Main Detection Interface ───────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run pothole detection on a BGR frame.
        Returns a list of detection dicts:
          {label, confidence, bbox, severity, guidance, steps, direction}
        """
        detections = []

        # ── Night mode check ─────────────────────────────────────────────
        is_night = self._is_night(frame)
        threshold = (config.YOLO_CONFIDENCE_NIGHT if is_night
                     else config.YOLO_CONFIDENCE)

        if is_night and not self._night_announced:
            detections.append({
                "label": "system",
                "confidence": 1.0,
                "message": "Low visibility mode active",
                "bbox": None,
                "severity": None,
                "guidance": None,
                "steps": None,
                "direction": None,
            })
            self._night_announced = True
        elif not is_night:
            self._night_announced = False

        # ── Apply night preprocessing ─────────────────────────────────────
        proc_frame = self._preprocess_night(frame) if is_night else frame

        # ── Primary: TFLite model ─────────────────────────────────────────
        if self._model_available:
            raw = self._run_model(proc_frame, threshold)
            detections.extend(raw)
        else:
            # ── Fallback: OpenCV texture analysis ─────────────────────────
            raw = self._opencv_fallback(proc_frame, threshold)
            detections.extend(raw)

        # ── Enrich each detection ─────────────────────────────────────────
        for det in detections:
            if det.get("bbox") is not None:
                det.update(self._enrich(det, frame))

        return detections

    # ── TFLite Inference ──────────────────────────────────────────────────
    def _run_model(self, frame: np.ndarray, threshold: float) -> list[dict]:
        results = []
        resized = cv2.resize(frame, self._input_shape)
        inp = np.expand_dims(resized.astype(np.uint8), axis=0)

        self._interp.set_tensor(self._input_idx, inp)
        self._interp.invoke()

        boxes   = self._interp.get_tensor(self._output_boxes)[0]
        classes = self._interp.get_tensor(self._output_classes)[0]
        scores  = self._interp.get_tensor(self._output_scores)[0]

        h, w = frame.shape[:2]
        for i, score in enumerate(scores):
            if score < threshold:
                continue
            cls_id = int(classes[i])
            if cls_id >= len(POTHOLE_CLASSES):
                continue
            label = POTHOLE_CLASSES[cls_id]
            y1, x1, y2, x2 = boxes[i]
            bbox = (
                int(x1 * w), int(y1 * h),
                int(x2 * w), int(y2 * h),
            )
            results.append({
                "label": label,
                "confidence": float(score),
                "bbox": bbox,
            })
        return results

    # ── OpenCV Fallback ───────────────────────────────────────────────────
    def _opencv_fallback(self, frame: np.ndarray, threshold: float) -> list[dict]:
        """
        Analyse the lower 40% of the frame for pothole-like texture:
        - Irregular edges (Canny)
        - Non-reflective / dark regions
        - Closed contours in ground zone
        """
        h, w = frame.shape[:2]
        ground_y = int(h * 0.60)
        ground_zone = frame[ground_y:, :]

        gray = cv2.cvtColor(ground_zone, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # tiny noise — skip
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Check if roughly convex (potholes are roughly elliptical)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.3:  # very irregular — skip
                continue

            # Check mean brightness in this region (potholes are darker)
            roi_mean = np.mean(gray[y:y+ch, x:x+cw])
            if roi_mean > 100:   # too bright — probably road, not pothole
                continue

            # Map back to full frame coords
            abs_bbox = (x, ground_y + y, x + cw, ground_y + y + ch)
            results.append({
                "label": "pothole",
                "confidence": min(0.6, threshold + 0.1),  # estimated
                "bbox": abs_bbox,
            })
        return results

    # ── Enrichment ────────────────────────────────────────────────────────
    def _enrich(self, det: dict, frame: np.ndarray) -> dict:
        """Add severity, guidance, steps, direction to a raw detection."""
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        frame_area = h * w
        obj_area   = (x2 - x1) * (y2 - y1)
        area_frac  = obj_area / frame_area

        # Severity
        if area_frac < config.POTHOLE_SMALL_AREA:
            severity = "small"
        elif area_frac < config.POTHOLE_MEDIUM_AREA:
            severity = "medium"
        else:
            severity = "large"

        # Depth estimate
        depth = estimate_distance(bbox, w, h)
        steps = depth.get("steps")
        direction = depth.get("direction", "directly ahead")

        # Check for water-filled (specular highlight in ROI)
        water_filled = self._check_water_filled(frame, bbox)
        label = det["label"]
        if water_filled:
            label = "water-filled pothole"

        # Guidance message
        cx = (x1 + x2) / 2
        frame_mid = w / 2
        if cx < w / 3:
            guidance = "Pothole on left, keep right"
        elif cx > 2 * w / 3:
            guidance = "Pothole on right, keep left"
        else:
            # Choose clearer side
            left_clear  = np.mean(frame[:, :int(w * 0.4)])
            right_clear = np.mean(frame[:, int(w * 0.6):])
            if left_clear > right_clear:
                guidance = "Pothole ahead, veer left"
            else:
                guidance = "Pothole ahead, veer right"

        # Build TTS message
        size_word = {"small": "small", "medium": "", "large": "large"}.get(severity, "")
        if size_word:
            size_word += " "
        if severity == "large":
            message = f"Large hazard ahead, stop and navigate around"
        elif severity == "medium":
            msg_label = label.replace("_", " ")
            message = (f"{msg_label} ahead, step carefully. "
                       f"{steps} steps, {direction}" if steps else
                       f"{msg_label} ahead, step carefully")
        else:
            msg_label = label.replace("_", " ")
            message = (f"Small {msg_label} ahead. {steps} steps, {direction}"
                       if steps else f"Small {msg_label} ahead")

        return {
            "severity": severity,
            "guidance": guidance,
            "message": message,
            "steps": steps,
            "direction": direction,
            "water_filled": water_filled,
        }

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _is_night(frame: np.ndarray) -> bool:
        """True if mean pixel brightness below threshold."""
        mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return mean_brightness < config.NIGHT_MODE_BRIGHTNESS

    @staticmethod
    def _preprocess_night(frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE + histogram equalisation for low-light enhancement."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _check_water_filled(frame: np.ndarray, bbox: tuple) -> bool:
        """
        Detect specular highlights inside bbox — indicator of water-filled pothole.
        Looks for bright, near-white patches (reflective surface).
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_frac = np.sum(thresh > 0) / thresh.size
        return bool(bright_frac > 0.15)   # >15% bright pixels = likely reflective water

    @staticmethod
    def detect_road_cracks(frame: np.ndarray) -> list:
        """
        Secondary: use Hough line transform on the ground zone to find road cracks.
        Returns list of line segments as (x1,y1,x2,y2).
        """
        h, w = frame.shape[:2]
        ground_zone = frame[int(h * 0.55):, :]
        gray = cv2.cvtColor(ground_zone, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30,
            minLineLength=40, maxLineGap=10)
        if lines is None:
            return []
        return [tuple(l[0]) for l in lines]


# ── Singleton ─────────────────────────────────────────────────────────────────
pothole_detector = PotholeDetector()
