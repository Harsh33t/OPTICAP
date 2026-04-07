"""
main_controller.py — OPTICap Master Orchestrator
Team Overclocked Minds | B.Tech CSE 2026

Starts all engines in correct order, wires callbacks, handles shutdown.
"""

import sys
import time
import signal
import logging
import os
import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration (must be first)
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.SYSTEM_LOG),
    ],
)
logger = logging.getLogger("opticap.main")

# ─────────────────────────────────────────────────────────────────────────────
# Import all engines (import order matters for singletons)
# ─────────────────────────────────────────────────────────────────────────────
from alert_queue      import alert_manager
from tts_engine       import tts_engine
from haptic_engine    import haptic_engine, buzzer_ctrl
from vision_engine    import vision_engine
from pothole_detector import pothole_detector
from face_recognition_engine import face_engine
from audio_engine     import audio_engine
from gesture_engine   import gesture_engine
from ultrasonic_engine import ultrasonic_engine
from button_handler   import button_handler
from ocr_engine       import ocr_engine


class OPTICap:
    """Master controller — boots all subsystems and manages lifecycle."""

    def __init__(self):
        self._shutdown_event = False

    # ── Bootstrap ─────────────────────────────────────────────────────────
    def setup(self):
        logger.info("═══ OPTICap Boot Sequence ═══")

        # 1. TTS Engine (must be first — other engines call speak)
        tts_engine.start()

        # 2. Alert Queue (wires TTS, haptic, buzzer)
        alert_manager.set_tts(tts_engine.speak)
        alert_manager.set_haptic(haptic_engine.play)
        alert_manager.set_buzzer(buzzer_ctrl.play)
        alert_manager.start()

        # 3. Hardware GPIO
        haptic_engine.setup()
        buzzer_ctrl.setup()
        ultrasonic_engine.setup()

        # 4. Vision
        vision_ok = vision_engine.load_model()
        if not vision_ok:
            logger.warning("YOLOv8 model not loaded — vision engine in passthrough mode.")
        pothole_detector.load_model()
        vision_engine.open_camera()

        # 5. Face & OCR engines
        face_engine.setup()
        ocr_engine.setup()

        # 6. Audio engine
        audio_engine.load_model()
        audio_engine.open_microphone()

        # 7. Gesture — NOT started yet (activated on demand by Button 3)
        gesture_engine.load_model()

        # 8. Buttons
        button_handler.set_mode_change_callback(self._on_mode_change)
        button_handler.set_sos_callback(self._on_sos)
        button_handler.set_gesture_toggle_callback(self._on_gesture_toggle)
        button_handler.set_battery_read_callback(self._on_battery_read)
        button_handler.set_speak(alert_manager.speak)
        button_handler.setup()

        logger.info("═══ Setup complete. Starting engines. ═══")

    # ── Start ─────────────────────────────────────────────────────────────
    def start(self):
        mode = config.MODE

        if mode in (config.MODE_VISION, config.MODE_ALL):
            vision_engine.start()
            face_engine.start(vision_engine.get_frame)
            ultrasonic_engine.start()

        if mode in (config.MODE_DEAF, config.MODE_ALL):
            audio_engine.start()

        # Announce startup
        button_handler.announce_startup()

        # Install signal handlers
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("OPTICap running. Press Ctrl+C to stop.")

        # ── Keep main thread alive ─────────────────────────────────────
        while not self._shutdown_event:
            time.sleep(0.5)

    # ── Graceful Shutdown ─────────────────────────────────────────────────
    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum} received — shutting down…")
        self._shutdown_event = True
        self.shutdown()

    def shutdown(self):
        logger.info("═══ OPTICap Shutdown ═══")
        alert_manager.speak("OPTICap shutting down", priority=config.PRIORITY_LOW)
        time.sleep(1.5)

        vision_engine.stop()
        face_engine.stop()
        audio_engine.stop()
        ultrasonic_engine.stop()

        if gesture_engine.is_active():
            gesture_engine.deactivate()

        button_handler.cleanup()
        ultrasonic_engine.cleanup()
        haptic_engine.cleanup()
        buzzer_ctrl.cleanup()
        alert_manager.stop()
        tts_engine.stop()

        logger.info("OPTICap halted.")
        sys.exit(0)

    # ── Mode Change Handler ───────────────────────────────────────────────
    def _on_mode_change(self, new_mode: str):
        """Reconfigure active engines when mode changes."""
        logger.info(f"Mode → {new_mode}")

        # Stop all engines first
        vision_engine.stop()
        face_engine.stop()
        audio_engine.stop()
        ultrasonic_engine.stop()

        time.sleep(0.3)   # Brief pause to let threads exit

        # Restart per new mode
        if new_mode in (config.MODE_VISION, config.MODE_ALL):
            vision_engine.open_camera()
            vision_engine.start()
            face_engine.start(vision_engine.get_frame)
            ultrasonic_engine.start()

        if new_mode in (config.MODE_DEAF, config.MODE_ALL):
            audio_engine.open_microphone()
            audio_engine.start()

        if new_mode == config.MODE_MUTE:
            # Gesture is toggled separately via Button 3
            pass

    # ── SOS Handler ───────────────────────────────────────────────────────
    def _on_sos(self):
        """Short press SOS — announce GPS or read OCR."""
        alert_manager.speak(
            "SOS pressed. GPS not connected. Reading surroundings.",
            priority=config.PRIORITY_MEDIUM, source="button")
        # Trigger OCR on current frame
        self._read_frame_text()

    def _read_frame_text(self):
        """OCR on current camera frame using AI."""
        frame = vision_engine.get_frame()
        if frame is None:
            return
        
        # Define callback to funnel OCR output straight into TTS
        def _on_ocr_result(text):
            alert_manager.speak(text, priority=config.PRIORITY_LOW, source="ocr")
            
        ocr_engine.read_frame_async(frame, _on_ocr_result)

    # ── Gesture Toggle ────────────────────────────────────────────────────
    def _on_gesture_toggle(self):
        if gesture_engine.is_active():
            gesture_engine.deactivate()
        else:
            gesture_engine.activate(config.CAMERA_INDEX)

    # ── Battery Read ──────────────────────────────────────────────────────
    def _on_battery_read(self):
        """
        Read battery voltage via ADC (optional ADS1115 on I2C).
        Falls back to a placeholder message.
        """
        try:
            import smbus2
            # ADS1115 @ 0x48: read channel 0 (voltage divider output)
            # Actual implementation depends on voltage divider ratio
            alert_manager.speak("Battery reading not configured", source="battery",
                                priority=config.PRIORITY_LOW)
        except ImportError:
            alert_manager.speak("Battery level: unknown. ADC not connected.",
                                source="battery", priority=config.PRIORITY_LOW)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    controller = OPTICap()
    controller.setup()
    controller.start()
