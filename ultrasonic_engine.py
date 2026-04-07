"""
ultrasonic_engine.py — OPTICap Dual HC-SR04 Proximity Engine
Team Overclocked Minds | B.Tech CSE 2026

Reads both ultrasonic sensors every 200ms in parallel threads.
Translates distance → haptic + TTS alerts with ultrasonic suppression.
"""

import time
import threading
import logging
from typing import Optional
import config
from alert_queue import alert_manager
from haptic_engine import haptic_engine, buzzer_ctrl
from vision_engine import vision_engine

logger = logging.getLogger("opticap.ultrasonic")

# ─────────────────────────────────────────────────────────────────────────────
# GPIO import with fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    _HW_GPIO = True
    logger.info("UltrasonicEngine using real RPi.GPIO.")
except ImportError:
    from mock_gpio import MockGPIO as GPIO
    _HW_GPIO = False
    logger.warning("RPi.GPIO not available — using MockGPIO for testing.")


class SensorThread(threading.Thread):
    """
    A dedicated thread for one HC-SR04 sensor.
    Publishes alerts whenever distance changes alert tier.
    """

    SPEED_OF_SOUND_CM_S = 34300.0
    MAX_TIMEOUT_SEC = 0.04   # 400cm / (34300 cm/s) ≈ 0.023s; use 40ms as safety

    def __init__(self, side: str, trig_pin: int, echo_pin: int):
        super().__init__(name=f"Ultrasonic-{side}", daemon=True)
        self._side     = side    # "left" or "right"
        self._trig     = trig_pin
        self._echo     = echo_pin
        self._running  = False
        self._last_alert_tier: int = 0    # 0=clear, 1=soft, 2=medium, 3=critical
        self._last_alert_time: float = 0.0

    def setup(self):
        GPIO.setup(self._trig, GPIO.OUT)
        GPIO.setup(self._echo, GPIO.IN)
        GPIO.output(self._trig, GPIO.LOW)
        time.sleep(0.05)  # sensor warm-up
        logger.info(f"Ultrasonic sensor ({self._side}) configured: "
                    f"TRIG={self._trig}, ECHO={self._echo}")

    def run(self):
        self._running = True
        while self._running:
            dist_cm = self._measure()
            if dist_cm is not None:
                self._evaluate(dist_cm)
            time.sleep(config.ULTRASONIC_POLL_INTERVAL)

    def stop(self):
        self._running = False

    # ── Distance Measurement ──────────────────────────────────────────────
    def _measure(self) -> Optional[float]:
        """
        Send 10µs trigger pulse, measure ECHO pulse duration.
        Returns distance in cm, or None on timeout/error.
        """
        try:
            GPIO.output(self._trig, GPIO.HIGH)
            time.sleep(0.00001)   # 10 µs
            GPIO.output(self._trig, GPIO.LOW)

            deadline = time.time() + self.MAX_TIMEOUT_SEC

            # Wait for echo to go HIGH
            while GPIO.input(self._echo) == GPIO.LOW:
                if time.time() > deadline:
                    return None
            pulse_start = time.time()

            # Wait for echo to go LOW
            while GPIO.input(self._echo) == GPIO.HIGH:
                if time.time() > deadline:
                    return None
            pulse_end = time.time()

            duration = pulse_end - pulse_start
            distance = (duration * self.SPEED_OF_SOUND_CM_S) / 2.0
            # Clamp to sensor range
            if distance < 2 or distance > 400:
                return None
            return round(distance, 1)

        except Exception as e:
            logger.error(f"Ultrasonic ({self._side}) measure error: {e}")
            return None

    # ── Alert Evaluation ───────────────────────────────────────────────────
    def _evaluate(self, dist_cm: float):
        """Map distance to alert tier and fire if tier changed or critical."""
        # Suppress if same-direction camera alert was recent
        if self._camera_alert_active():
            return

        now = time.time()
        tier = self._distance_to_tier(dist_cm)

        # Always alert on CRITICAL, otherwise rate-limit transitions
        min_gap = 1.5   # seconds between non-critical alerts
        if tier == self._last_alert_tier and tier != 3:
            return
        if tier < self._last_alert_tier and now - self._last_alert_time < min_gap:
            return

        self._last_alert_tier = tier
        if tier == 0:
            return   # Obstacle cleared — no announcement

        self._last_alert_time = now

        if tier == 3:
            # CRITICAL
            msg = f"Obstacle on {self._side}, very close!"
            alert_manager.speak(msg, priority=config.PRIORITY_CRITICAL, source="ultrasonic")
            buzzer_ctrl.play("rapid_triple")
        elif tier == 2:
            # Moderate vibration on corresponding side
            haptic_engine.pulse(self._side, duration_ms=300)
        elif tier == 1:
            # Soft single vibration
            haptic_engine.pulse(self._side, duration_ms=100)

    def _distance_to_tier(self, dist_cm: float) -> int:
        if dist_cm < config.ULTRASONIC_CRITICAL_CM:
            return 3
        if dist_cm < config.ULTRASONIC_MEDIUM_CM:
            return 2
        if dist_cm < config.ULTRASONIC_SOFT_CM:
            return 1
        return 0

    def _camera_alert_active(self) -> bool:
        """Check if vision_engine fired an alert for this direction recently."""
        if not hasattr(vision_engine, "last_camera_alert_dir"):
            return False
        same_dir = (vision_engine.last_camera_alert_dir in self._side or
                    self._side in vision_engine.last_camera_alert_dir)
        recent = (time.time() - vision_engine.last_camera_alert_time <
                  config.ULTRASONIC_SUPPRESS_SEC)
        return same_dir and recent


# ─────────────────────────────────────────────────────────────────────────────
# Ultrasonic Engine Controller
# ─────────────────────────────────────────────────────────────────────────────
class UltrasonicEngine:
    def __init__(self):
        self._left_sensor  = SensorThread(
            "left", config.ULTRASONIC_LEFT_TRIG, config.ULTRASONIC_LEFT_ECHO)
        self._right_sensor = SensorThread(
            "right", config.ULTRASONIC_RIGHT_TRIG, config.ULTRASONIC_RIGHT_ECHO)

    def setup(self):
        self._left_sensor.setup()
        self._right_sensor.setup()

    def start(self):
        self._left_sensor.start()
        self._right_sensor.start()
        logger.info("UltrasonicEngine started (both sensors).")

    def stop(self):
        self._left_sensor.stop()
        self._right_sensor.stop()
        self._left_sensor.join(timeout=1.0)
        self._right_sensor.join(timeout=1.0)
        logger.info("UltrasonicEngine stopped.")

    def cleanup(self):
        try:
            GPIO.cleanup([
                config.ULTRASONIC_LEFT_TRIG,  config.ULTRASONIC_LEFT_ECHO,
                config.ULTRASONIC_RIGHT_TRIG, config.ULTRASONIC_RIGHT_ECHO,
            ])
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────
ultrasonic_engine = UltrasonicEngine()
