"""
button_handler.py — OPTICap GPIO Button Handler & Mode Controller
Team Overclocked Minds | B.Tech CSE 2026

3 fabric-mounted push buttons with short-press and long-press actions.
Uses GPIO edge detection (interrupt-driven), not polling.
"""

import time
import threading
import logging
from typing import Optional, Callable
import config

logger = logging.getLogger("opticap.buttons")

# ─────────────────────────────────────────────────────────────────────────────
# GPIO import with fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    _HW_GPIO = True
except ImportError:
    from mock_gpio import MockGPIO as GPIO
    _HW_GPIO = False
    logger.warning("ButtonHandler using MockGPIO.")


# ─────────────────────────────────────────────────────────────────────────────
# Button Handler
# ─────────────────────────────────────────────────────────────────────────────
class ButtonHandler:
    """
    Manages 3 push buttons with debounce, short-press, and long-press detection.

    Exposes:
      on_mode_short_press()       → cycle modes
      on_mode_long_press()        → mute all alerts
      on_sos_short_press()        → speak GPS / OCR
      on_sos_long_press()         → SOS alert
      on_gesture_short_press()    → toggle gesture engine
      on_gesture_long_press()     → read battery level
    """

    DEBOUNCE_MS        = config.BUTTON_DEBOUNCE_MS
    LONG_PRESS_SEC     = config.BUTTON_LONG_PRESS_SEC

    def __init__(self):
        self._press_times: dict[int, float] = {}
        self._debounce_times: dict[int, float] = {}

        # Current device mode
        self._mode_idx = config.MODES_ORDER.index(config.MODE)

        # Mute state
        self._muted = False

        # Callbacks — injected by main_controller
        self._on_mode_change:   Optional[Callable[[str], None]] = None
        self._on_sos:           Optional[Callable[[], None]]    = None
        self._on_gesture_toggle:Optional[Callable[[], None]]    = None
        self._on_battery_read:  Optional[Callable[[], None]]    = None
        self._speak_fn:         Optional[Callable[[str, int], None]] = None

    # ── Dependency Injection ──────────────────────────────────────────────
    def set_mode_change_callback(self, fn: Callable[[str], None]):
        self._on_mode_change = fn

    def set_sos_callback(self, fn: Callable[[], None]):
        self._on_sos = fn

    def set_gesture_toggle_callback(self, fn: Callable[[], None]):
        self._on_gesture_toggle = fn

    def set_battery_read_callback(self, fn: Callable[[], None]):
        self._on_battery_read = fn

    def set_speak(self, fn: Callable[[str, int], None]):
        self._speak_fn = fn

    # ── Setup ─────────────────────────────────────────────────────────────
    def setup(self):
        for pin in [config.BUTTON_MODE, config.BUTTON_SOS, config.BUTTON_GESTURE]:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                pin, GPIO.BOTH,
                callback=self._gpio_callback,
                bouncetime=self.DEBOUNCE_MS,
            )
        logger.info("Button GPIO configured.")

    def cleanup(self):
        for pin in [config.BUTTON_MODE, config.BUTTON_SOS, config.BUTTON_GESTURE]:
            try:
                GPIO.remove_event_detect(pin)
            except Exception:
                pass
        try:
            GPIO.cleanup([config.BUTTON_MODE, config.BUTTON_SOS, config.BUTTON_GESTURE])
        except Exception:
            pass

    # ── GPIO Interrupt Callback ───────────────────────────────────────────
    def _gpio_callback(self, pin: int):
        """
        Called on BOTH edges. Track press/release times to distinguish
        short vs long press. Buttons are active LOW (PUD_UP).
        """
        now = time.time()

        # Debounce
        last_db = self._debounce_times.get(pin, 0.0)
        if now - last_db < self.DEBOUNCE_MS / 1000.0:
            return
        self._debounce_times[pin] = now

        state = GPIO.input(pin)

        if state == GPIO.LOW:
            # Button pressed ↓
            self._press_times[pin] = now

        elif state == GPIO.HIGH:
            # Button released ↑
            pressed_at = self._press_times.get(pin)
            if pressed_at is None:
                return
            duration = now - pressed_at
            self._press_times.pop(pin, None)

            if duration >= self.LONG_PRESS_SEC:
                self._dispatch_long(pin)
            else:
                self._dispatch_short(pin)

    # ── Dispatch ──────────────────────────────────────────────────────────
    def _dispatch_short(self, pin: int):
        logger.info(f"Short press: pin={pin}")
        if pin == config.BUTTON_MODE:
            self._cycle_mode()
        elif pin == config.BUTTON_SOS:
            self._sos_short()
        elif pin == config.BUTTON_GESTURE:
            self._gesture_short()

    def _dispatch_long(self, pin: int):
        logger.info(f"Long press: pin={pin}")
        if pin == config.BUTTON_MODE:
            self._toggle_mute()
        elif pin == config.BUTTON_SOS:
            self._sos_long()
        elif pin == config.BUTTON_GESTURE:
            self._battery_read()

    # ── Actions ───────────────────────────────────────────────────────────
    def _cycle_mode(self):
        self._mode_idx = (self._mode_idx + 1) % len(config.MODES_ORDER)
        new_mode = config.MODES_ORDER[self._mode_idx]
        config.MODE = new_mode
        msg = f"{new_mode.capitalize()} mode active"
        logger.info(f"Mode changed to: {new_mode}")
        self._speak(msg, config.PRIORITY_MEDIUM)
        if self._on_mode_change:
            self._on_mode_change(new_mode)

    def _toggle_mute(self):
        self._muted = not self._muted
        if self._muted:
            self._speak("All alerts muted", config.PRIORITY_MEDIUM)
            logger.info("Alerts muted.")
        else:
            self._speak("Alerts resumed", config.PRIORITY_MEDIUM)
            logger.info("Alerts resumed.")

    def _sos_short(self):
        logger.info("SOS short press — read GPS / OCR")
        if self._on_sos:
            self._on_sos()
        else:
            self._speak("SOS function not configured", config.PRIORITY_MEDIUM)

    def _sos_long(self):
        logger.info("SOS long press — trigger SOS sequence")
        self._speak("SOS alert activated. Sending emergency signal.", config.PRIORITY_CRITICAL)
        # Buzzer SOS pattern runs from haptic_engine, triggered by alert_manager
        from alert_queue import alert_manager
        alert_manager.buzzer("sos", priority=config.PRIORITY_CRITICAL, source="button")

    def _gesture_short(self):
        logger.info("Gesture toggle")
        if self._on_gesture_toggle:
            self._on_gesture_toggle()

    def _battery_read(self):
        logger.info("Battery read request")
        if self._on_battery_read:
            self._on_battery_read()
        else:
            self._speak("Battery monitoring not configured", config.PRIORITY_LOW)

    def _speak(self, text: str, priority: int = config.PRIORITY_LOW):
        if self._speak_fn and not self._muted:
            self._speak_fn(text, priority)

    # ── Public Properties ─────────────────────────────────────────────────
    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def current_mode(self) -> str:
        return config.MODES_ORDER[self._mode_idx]

    # ── Startup Announcement ──────────────────────────────────────────────
    def announce_startup(self):
        mode = self.current_mode
        msg = f"OPTICap ready. {mode.capitalize()} mode active."
        self._speak(msg, config.PRIORITY_MEDIUM)


# ── Singleton ─────────────────────────────────────────────────────────────────
button_handler = ButtonHandler()
