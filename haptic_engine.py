"""
haptic_engine.py — OPTICap Vibration Motor Controller
Team Overclocked Minds | B.Tech CSE 2026

Controls two 10mm coin vibration motors (left / right) via GPIO PWM.
Pattern definitions are driven by config.HAPTIC_PATTERNS.
"""

import threading
import time
import logging
import config

logger = logging.getLogger("opticap.haptic")

# ─────────────────────────────────────────────────────────────────────────────
# GPIO abstraction — falls back to MockGPIO when RPi.GPIO unavailable
# ─────────────────────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    _HW_GPIO = True
    logger.info("Haptic engine using real RPi.GPIO.")
except ImportError:
    from mock_gpio import MockGPIO as GPIO  # noqa: F401 — defined in mock_gpio.py
    _HW_GPIO = False
    logger.warning("RPi.GPIO not available. Using MockGPIO.")


class HapticEngine:
    """
    Drive left and right vibration motors with various patterns.
    Motor pins are defined in config.py (MOTOR_LEFT, MOTOR_RIGHT).
    """

    MOTOR_FREQ = 100   # PWM frequency in Hz

    def __init__(self):
        self._left_pin  = config.MOTOR_LEFT
        self._right_pin = config.MOTOR_RIGHT
        self._left_pwm  = None
        self._right_pwm = None
        self._lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def setup(self):
        GPIO.setup(self._left_pin,  GPIO.OUT)
        GPIO.setup(self._right_pin, GPIO.OUT)
        self._left_pwm  = GPIO.PWM(self._left_pin,  self.MOTOR_FREQ)
        self._right_pwm = GPIO.PWM(self._right_pin, self.MOTOR_FREQ)
        self._left_pwm.start(0)
        self._right_pwm.start(0)
        logger.info("HapticEngine GPIO configured.")

    def cleanup(self):
        if self._left_pwm:
            self._left_pwm.stop()
        if self._right_pwm:
            self._right_pwm.stop()
        logger.info("HapticEngine cleaned up.")

    # ── Public API ────────────────────────────────────────────────────────
    def play(self, haptic_data: dict, side: str = "both"):
        """
        Execute a haptic pattern asynchronously.
        `haptic_data` matches a config.HAPTIC_PATTERNS entry.
        `side` = 'left' | 'right' | 'both'
        """
        t = threading.Thread(
            target=self._execute, args=(haptic_data, side), daemon=True)
        t.start()

    def pulse(self, side: str, duration_ms: int = 300, duty: int = 90):
        """Simple single pulse on one or both motors."""
        data = {"pattern": "single_pulse", "cycles": 1, "duration_ms": duration_ms}
        t = threading.Thread(
            target=self._execute, args=(data, side), daemon=True)
        t.start()

    def stop_all(self):
        with self._lock:
            self._set(0, 0)

    # ── Pattern Executor ──────────────────────────────────────────────────
    def _execute(self, data: dict, side: str):
        pattern  = data.get("pattern", "single_pulse")
        cycles   = data.get("cycles", 1)
        dur_ms   = data.get("duration_ms", 200)
        dur_s    = dur_ms / 1000.0
        gap_s    = 0.08  # standard inter-pulse gap

        with self._lock:
            for _ in range(cycles):
                if pattern == "single_pulse" or pattern == "single_burst":
                    self._pulse_side(side, dur_s, duty=90)

                elif pattern == "alternating_rapid":
                    self._pulse_side("left",  dur_s, duty=90)
                    time.sleep(gap_s)
                    self._pulse_side("right", dur_s, duty=90)
                    time.sleep(gap_s)

                elif pattern == "both_continuous":
                    self._set(90, 90)
                    time.sleep(dur_s)
                    self._set(0, 0)
                    time.sleep(gap_s)

                elif pattern == "long_center":
                    # Simultaneous on both = "center"
                    self._set(70, 70)
                    time.sleep(dur_s)
                    self._set(0, 0)

                else:
                    # Default: single pulse
                    self._pulse_side(side, dur_s, duty=80)

    def _pulse_side(self, side: str, duration: float, duty: int = 90):
        if side == "left":
            self._set(duty, 0)
        elif side == "right":
            self._set(0, duty)
        else:  # both
            self._set(duty, duty)
        time.sleep(duration)
        self._set(0, 0)

    def _set(self, left_duty: int, right_duty: int):
        if self._left_pwm:
            self._left_pwm.ChangeDutyCycle(left_duty)
        if self._right_pwm:
            self._right_pwm.ChangeDutyCycle(right_duty)


# ── Buzzer Controller ─────────────────────────────────────────────────────────
class BuzzerController:
    """
    Controls 5V active piezo buzzer on BUZZER_PIN.
    Patterns defined in config.py.
    """

    PATTERN_MAP = {
        "rapid_triple": config.BUZZER_RAPID_TRIPLE,
        "double":       config.BUZZER_DOUBLE,
        "single":       config.BUZZER_SINGLE,
        "sos":          config.BUZZER_SOS,
    }

    def __init__(self):
        self._pin = config.BUZZER_PIN
        self._lock = threading.Lock()

    def setup(self):
        GPIO.setup(self._pin, GPIO.OUT)
        GPIO.output(self._pin, GPIO.LOW)
        logger.info("BuzzerController GPIO configured.")

    def cleanup(self):
        GPIO.output(self._pin, GPIO.LOW)

    def play(self, pattern_name: str):
        """Execute a named buzzer pattern asynchronously."""
        t = threading.Thread(
            target=self._execute, args=(pattern_name,), daemon=True)
        t.start()

    def _execute(self, pattern_name: str):
        pattern = self.PATTERN_MAP.get(pattern_name, config.BUZZER_SINGLE)
        with self._lock:
            for on_sec, off_sec in pattern:
                GPIO.output(self._pin, GPIO.HIGH)
                time.sleep(on_sec)
                GPIO.output(self._pin, GPIO.LOW)
                if off_sec > 0:
                    time.sleep(off_sec)


# ── Singletons ────────────────────────────────────────────────────────────────
haptic_engine  = HapticEngine()
buzzer_ctrl    = BuzzerController()
