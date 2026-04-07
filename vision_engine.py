"""
vision_engine.py — OPTICap Vision Engine
Team Overclocked Minds | B.Tech CSE 2026

YOLOv8-Nano INT8 TFLite object detection at 10–15 FPS.
Implements the 4-tier Navigational Priority Matrix.
Traffic light color detection sub-module included.
"""

import cv2
import numpy as np
import time
import threading
import logging
import math
from collections import deque
from typing import Optional
import config
from depth_estimator import estimate_distance, describe_distance, _horizontal_direction
from pothole_detector import pothole_detector
from alert_queue import alert_manager, Alert, ALERT_TYPE_COMBO, ALERT_TYPE_SPEECH

logger = logging.getLogger("opticap.vision")

# ─────────────────────────────────────────────────────────────────────────────
# TFLite import
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
        logger.error("tflite_runtime not available! Vision engine will not detect objects.")

# ─────────────────────────────────────────────────────────────────────────────
# COCO class names that YOLOv8-Nano knows (indices 0-79)
# Trimmed to the ones we care about; full list loaded from labels file if present.
# ─────────────────────────────────────────────────────────────────────────────
COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train",
    "truck","boat","traffic light","fire hydrant","stop sign",
    "parking meter","bench","bird","cat","dog","horse","sheep",
    "cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv",
    "laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush",
]

# Extra India-specific labels (appended to COCO)
EXTRA_LABELS = [
    "auto_rickshaw","tuk_tuk","e-scooter","street_vendor_cart",
    "speed_bump","pothole","open_drain","construction_debris",
    "traffic_signal","pedestrian_crossing","garbage_pile",
    "electric_pole","low_hanging_wire","open_manhole",
    "wet_floor","road_depression","stairs_up","stairs_down",
    "ramp","curb","goat","door","manhole",
]

ALL_LABELS = COCO_LABELS + EXTRA_LABELS

# Build reverse lookup: name → index
LABEL_INDEX = {name: idx for idx, name in enumerate(ALL_LABELS)}

# ─────────────────────────────────────────────────────────────────────────────
# Priority tier maps (built at import time from config)
# ─────────────────────────────────────────────────────────────────────────────
_TIER1_SET  = set(config.PRIORITY_TIER_1["classes"])
_TIER2_SET  = set(config.PRIORITY_TIER_2["classes"])
_TIER3_SET  = set(config.PRIORITY_TIER_3["classes"])
_TIER4_SET  = set(config.PRIORITY_TIER_4["classes"])


def _get_tier(label: str) -> Optional[dict]:
    if label in _TIER1_SET:
        return config.PRIORITY_TIER_1
    if label in _TIER2_SET:
        return config.PRIORITY_TIER_2
    if label in _TIER3_SET:
        return config.PRIORITY_TIER_3
    if label in _TIER4_SET:
        return config.PRIORITY_TIER_4
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Vision Engine
# ─────────────────────────────────────────────────────────────────────────────
class VisionEngine:
    """
    Captures frames, runs YOLOv8-Nano TFLite inference, and publishes
    priority alerts to the AlertQueueManager.
    """

    def __init__(self):
        self._interp = None
        self._input_idx = None
        self._output_boxes = None
        self._output_classes = None
        self._output_scores = None
        self._input_shape = (320, 320)
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Deduplication: deque of (label, direction, timestamp)
        self._recent: deque = deque(maxlen=config.DEDUP_HISTORY_SIZE)
        self._cooldowns: dict = {}   # (label, direction) → last alert time

        # Last camera alert direction for ultrasonic suppression
        self.last_camera_alert_dir: str = ""
        self.last_camera_alert_time: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def load_model(self) -> bool:
        import os
        if not _TFLITE_OK:
            return False
        if not os.path.exists(config.YOLO_MODEL_PATH):
            logger.error(f"YOLOv8 model not found at {config.YOLO_MODEL_PATH}")
            return False
        try:
            self._interp = tflite.Interpreter(
                model_path=config.YOLO_MODEL_PATH,
                num_threads=4,
            )
            self._interp.allocate_tensors()
            input_details  = self._interp.get_input_details()
            output_details = self._interp.get_output_details()
            self._input_idx = input_details[0]["index"]
            # YOLOv8 outputs: [boxes, scores] or SSD style [boxes,classes,scores]
            self._output_boxes   = output_details[0]["index"]
            if len(output_details) > 2:
                self._output_classes = output_details[1]["index"]
                self._output_scores  = output_details[2]["index"]
            else:
                self._output_classes = None
                self._output_scores  = output_details[1]["index"]
            h, w = input_details[0]["shape"][1:3]
            self._input_shape = (int(w), int(h))
            logger.info(f"YOLOv8 TFLite model loaded. Input shape: {self._input_shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def open_camera(self) -> bool:
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            logger.error("Failed to open camera.")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
        logger.info("Camera opened.")
        return True

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="VisionEngine", daemon=True)
        self._thread.start()
        logger.info("VisionEngine started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info("VisionEngine stopped.")

    # ── Main Loop ─────────────────────────────────────────────────────────
    def _loop(self):
        frame_count = 0
        t_start = time.time()
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame grab failed.")
                time.sleep(0.1)
                continue

            frame_count += 1
            detections = self._infer(frame)
            self._process_detections(detections, frame)

            # Also run pothole detector on same frame
            pothole_dets = pothole_detector.detect(frame)
            self._process_pothole_dets(pothole_dets)

            # FPS tracking
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed
                logger.debug(f"Vision FPS: {fps:.1f}")

    # ── YOLOv8 Inference ─────────────────────────────────────────────────
    def _infer(self, frame: np.ndarray) -> list[dict]:
        if self._interp is None:
            return []
        try:
            resized = cv2.resize(frame, self._input_shape)
            inp = np.expand_dims(resized.astype(np.uint8), axis=0)
            self._interp.set_tensor(self._input_idx, inp)
            self._interp.invoke()

            boxes  = self._interp.get_tensor(self._output_boxes)[0]
            scores = self._interp.get_tensor(self._output_scores)[0]
            if self._output_classes is not None:
                classes = self._interp.get_tensor(self._output_classes)[0]
            else:
                # Single-output YOLOv8: shape [N, 6] where last col is class
                classes = boxes[:, 5].astype(int) if boxes.ndim == 2 else np.zeros(len(scores))
                if boxes.ndim == 2:
                    boxes = boxes[:, :4]

            threshold = self._current_threshold(frame)
            h, w = frame.shape[:2]
            results = []
            for i, score in enumerate(scores):
                if float(score) < threshold:
                    continue
                cls_id = int(classes[i]) if self._output_classes else int(classes[i])
                if cls_id >= len(ALL_LABELS):
                    continue
                label = ALL_LABELS[cls_id]
                # boxes in [y1,x1,y2,x2] normalised or pixel?
                b = boxes[i]
                if max(b) <= 1.0:
                    y1, x1, y2, x2 = b
                    bbox = (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
                else:
                    x1, y1, x2, y2 = b[:4]
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                results.append({"label": label, "confidence": float(score), "bbox": bbox})
            return results
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    def _current_threshold(self, frame: np.ndarray) -> float:
        mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if mean_brightness < config.NIGHT_MODE_BRIGHTNESS:
            return config.YOLO_CONFIDENCE_NIGHT
        return config.YOLO_CONFIDENCE

    # ── Detection Processing ──────────────────────────────────────────────
    def _process_detections(self, detections: list[dict], frame: np.ndarray):
        for det in detections:
            label = det["label"]
            bbox  = det["bbox"]
            tier  = _get_tier(label)
            if tier is None:
                continue

            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            cx = (x1 + x2) / 2.0
            direction = _horizontal_direction(cx, w)
            key = (label, direction)

            # Deduplication cooldown
            cooldown = tier["cooldown"]
            last = self._cooldowns.get(key, 0.0)
            if time.time() - last < cooldown:
                continue
            self._cooldowns[key] = time.time()

            # Record camera alert for ultrasonic suppression
            self.last_camera_alert_dir  = direction
            self.last_camera_alert_time = time.time()

            priority = tier["priority"]

            # ── Traffic light special handling ─────────────────────────
            if label in ("traffic_light", "traffic_signal"):
                tl_msg = detect_traffic_light_color(frame, bbox)
                if tl_msg:
                    alert_manager.speak(tl_msg, priority=config.PRIORITY_MEDIUM,
                                        source="vision")
                continue

            # ── Tier 1: Vehicle ────────────────────────────────────────
            if priority == config.PRIORITY_CRITICAL:
                alert_manager.combo(
                    message=tier["message"],
                    haptic_data={"pattern": "both_continuous", "cycles": 2, "duration_ms": 300},
                    buzzer_pattern=tier.get("buzzer_pattern", "rapid_triple"),
                    side="both",
                    priority=config.PRIORITY_CRITICAL,
                    source="vision",
                )

            # ── Tier 2: Ground hazard ─────────────────────────────────
            elif priority == config.PRIORITY_HIGH:
                depth = estimate_distance(bbox, w, h)
                steps_str = f"{depth['steps']} steps" if depth.get("steps") else ""
                msg = f"{label.replace('_',' ')} ahead"
                if steps_str:
                    msg += f", {steps_str}, {direction}"
                alert_manager.speak(msg, priority=config.PRIORITY_HIGH, source="vision")

            # ── Tier 3: Structural ────────────────────────────────────
            elif priority == config.PRIORITY_MEDIUM:
                label_str = label.replace("_", " ")
                msg = f"{label_str} {direction}"
                alert_manager.combo(
                    message=msg,
                    haptic_data={"pattern": "alternating_rapid", "cycles": 1, "duration_ms": 150},
                    buzzer_pattern="double",
                    side=_side_from_dir(direction),
                    priority=config.PRIORITY_MEDIUM,
                    source="vision",
                )

            # ── Tier 4: Informational ─────────────────────────────────
            else:
                label_str = label.replace("_", " ")
                msg = f"{label_str} {direction}"
                alert_manager.speak(msg, priority=config.PRIORITY_LOW, source="vision")

    def _process_pothole_dets(self, dets: list[dict]):
        for det in dets:
            if det.get("message"):
                alert_manager.speak(
                    det["message"],
                    priority=config.PRIORITY_HIGH,
                    source="pothole",
                )

    # ── Current Frame Access (for face engine etc.) ─────────────────────
    def get_frame(self) -> Optional[np.ndarray]:
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Traffic Light Color Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_traffic_light_color(frame: np.ndarray, bbox: tuple) -> Optional[str]:
    """
    Crop bounding box and apply HSV color masking to determine Red/Yellow/Green.
    Returns a speech string, or None if indeterminate.
    """
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def mask_frac(lower, upper):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return np.sum(mask > 0) / mask.size

    red_frac = (mask_frac(config.TL_RED_LOWER1, config.TL_RED_UPPER1) +
                mask_frac(config.TL_RED_LOWER2, config.TL_RED_UPPER2))
    green_frac  = mask_frac(config.TL_GREEN_LOWER,  config.TL_GREEN_UPPER)
    yellow_frac = mask_frac(config.TL_YELLOW_LOWER, config.TL_YELLOW_UPPER)

    best = max(red_frac, green_frac, yellow_frac)
    if best < 0.05:   # No clear dominant colour
        return "Traffic light detected"
    if best == red_frac:
        return "Red light ahead, stop"
    if best == green_frac:
        return "Green light, safe to cross"
    return "Yellow light ahead, slow down"


def _side_from_dir(direction: str) -> str:
    if "left" in direction:
        return "left"
    if "right" in direction:
        return "right"
    return "both"


# ── Singleton ─────────────────────────────────────────────────────────────────
vision_engine = VisionEngine()
