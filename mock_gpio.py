"""
mock_gpio.py — MockGPIO for CI testing without Raspberry Pi hardware
Team Overclocked Minds | B.Tech CSE 2026

Drop-in replacement for RPi.GPIO so all other modules can be
unit tested on any machine (Linux, macOS, Windows) without hardware.
"""

import logging
import time

logger = logging.getLogger("opticap.mock_gpio")

# ── Constants matching RPi.GPIO API ───────────────────────────────────────────
BCM  = "BCM"
BOARD = "BOARD"
OUT  = "OUT"
IN   = "IN"
HIGH = True
LOW  = False
BOTH = "BOTH"
RISING  = "RISING"
FALLING = "FALLING"
PUD_UP   = "PUD_UP"
PUD_DOWN = "PUD_DOWN"


class _MockPWM:
    def __init__(self, pin: int, frequency: int):
        self.pin = pin
        self.frequency = frequency
        self.duty = 0
        logger.debug(f"[MockGPIO] PWM set up on pin {pin} @ {frequency}Hz")

    def start(self, duty_cycle: int):
        self.duty = duty_cycle
        logger.debug(f"[MockGPIO] PWM pin={self.pin} start duty={duty_cycle}")

    def ChangeDutyCycle(self, duty_cycle: int):
        self.duty = duty_cycle
        logger.debug(f"[MockGPIO] PWM pin={self.pin} duty={duty_cycle}")

    def ChangeFrequency(self, frequency: int):
        self.frequency = frequency

    def stop(self):
        logger.debug(f"[MockGPIO] PWM pin={self.pin} stopped")


class MockGPIO:
    """
    Mimics the RPi.GPIO interface so all engine files can import
    `from mock_gpio import MockGPIO as GPIO` without error.
    """

    BCM   = BCM
    BOARD = BOARD
    OUT   = OUT
    IN    = IN
    HIGH  = HIGH
    LOW   = LOW
    BOTH  = BOTH
    RISING   = RISING
    FALLING  = FALLING
    PUD_UP   = PUD_UP
    PUD_DOWN = PUD_DOWN

    _pin_states: dict = {}
    _callbacks:  dict = {}
    _mode: str = BCM

    @classmethod
    def setmode(cls, mode: str):
        cls._mode = mode
        logger.debug(f"[MockGPIO] Mode set to {mode}")

    @classmethod
    def setwarnings(cls, flag: bool):
        pass

    @classmethod
    def setup(cls, pin: int, direction, pull_up_down=None, initial=None):
        cls._pin_states[pin] = LOW if direction == OUT else HIGH
        logger.debug(f"[MockGPIO] setup pin={pin} dir={direction}")

    @classmethod
    def output(cls, pin: int, value):
        cls._pin_states[pin] = value
        logger.debug(f"[MockGPIO] output pin={pin} val={value}")

    @classmethod
    def input(cls, pin: int):
        val = cls._pin_states.get(pin, LOW)
        logger.debug(f"[MockGPIO] input pin={pin} → {val}")
        return val

    @classmethod
    def PWM(cls, pin: int, frequency: int) -> _MockPWM:
        return _MockPWM(pin, frequency)

    @classmethod
    def add_event_detect(cls, pin: int, edge, callback=None, bouncetime=200):
        cls._callbacks[pin] = callback
        logger.debug(
            f"[MockGPIO] event detect pin={pin} edge={edge} bounce={bouncetime}ms")

    @classmethod
    def remove_event_detect(cls, pin: int):
        cls._callbacks.pop(pin, None)

    @classmethod
    def simulate_button_press(cls, pin: int):
        """Helper for unit tests: simulate an edge event on a pin."""
        cb = cls._callbacks.get(pin)
        if cb:
            cb(pin)

    @classmethod
    def cleanup(cls, pins=None):
        if pins is None:
            cls._pin_states.clear()
        else:
            for p in (pins if hasattr(pins, "__iter__") else [pins]):
                cls._pin_states.pop(p, None)
        logger.debug("[MockGPIO] cleanup called")


# ── Allow `import mock_gpio; mock_gpio.GPIO` style ─────────────────────────
GPIO = MockGPIO
