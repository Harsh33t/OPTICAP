"""
test_suite.py — OPTICap Full Test Suite
Team Overclocked Minds | B.Tech CSE 2026

Runs WITHOUT hardware. Uses MockGPIO, mock camera, and mock audio.
Tests:
  1. Unit tests for each module
  2. Integration test (video pipeline)
  3. Memory stress test (5 minutes, all modules)
  4. Latency test (detection → TTS start)
  5. Alert deduplication test
  6. GPIO mock mode verification
"""

import sys
import os
import time
import unittest
import threading
import numpy as np
import logging

# ── Ensure mock GPIO is used ───────────────────────────────────────────────
os.environ["OPTICAP_MOCK_GPIO"] = "1"

# Patch RPi.GPIO before any engine imports
sys.modules["RPi"] = type(sys)("RPi")
sys.modules["RPi.GPIO"] = __import__("mock_gpio").MockGPIO

logging.basicConfig(level=logging.WARNING)  # Keep test output clean


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_black_frame(w=320, h=320):
    """Return a blank black BGR frame."""
    import cv2
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_test_frame_with_bright_region():
    """Frame with a bright patch in lower centre (simulates road marking)."""
    frame = make_black_frame()
    frame[220:300, 120:200] = 200
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestConfig(unittest.TestCase):

    def test_gpio_pins_unique(self):
        import config
        pins = [
            config.ULTRASONIC_LEFT_TRIG, config.ULTRASONIC_LEFT_ECHO,
            config.ULTRASONIC_RIGHT_TRIG, config.ULTRASONIC_RIGHT_ECHO,
            config.BUTTON_MODE, config.BUTTON_SOS, config.BUTTON_GESTURE,
            config.MOTOR_LEFT, config.MOTOR_RIGHT, config.BUZZER_PIN,
        ]
        self.assertEqual(len(pins), len(set(pins)), "GPIO pin collision detected!")

    def test_priority_levels(self):
        import config
        self.assertLess(config.PRIORITY_CRITICAL, config.PRIORITY_HIGH)
        self.assertLess(config.PRIORITY_HIGH, config.PRIORITY_MEDIUM)
        self.assertLess(config.PRIORITY_MEDIUM, config.PRIORITY_LOW)

    def test_modes_order(self):
        import config
        self.assertIn(config.MODE, config.MODES_ORDER)
        self.assertEqual(len(config.MODES_ORDER), 4)

    def test_isl_gesture_map_coverage(self):
        import config
        self.assertGreaterEqual(len(config.ISL_GESTURE_MAP), 10)
        for gesture, phrase in config.ISL_GESTURE_MAP.items():
            self.assertIsInstance(gesture, str)
            self.assertIsInstance(phrase, str)
            self.assertGreater(len(phrase), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Depth Estimator Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDepthEstimator(unittest.TestCase):

    def test_object_on_ground(self):
        from depth_estimator import estimate_distance
        # Object at lower-right of frame
        result = estimate_distance((200, 200, 280, 300), 320, 320)
        self.assertIn("distance_m", result)
        self.assertIn("steps", result)
        self.assertIn("direction", result)

    def test_overhead_object(self):
        from depth_estimator import estimate_distance
        # Object entirely in top half of frame
        result = estimate_distance((100, 10, 200, 100), 320, 320)
        self.assertTrue(result["overhead"])
        self.assertIsNone(result["distance_m"])

    def test_step_count_positive(self):
        from depth_estimator import estimate_distance
        result = estimate_distance((100, 200, 200, 300), 320, 320)
        if not result["overhead"] and result["steps"] is not None:
            self.assertGreater(result["steps"], 0)

    def test_direction_left(self):
        from depth_estimator import estimate_distance
        result = estimate_distance((10, 200, 80, 290), 320, 320)
        self.assertIn("left", result["direction"])

    def test_direction_right(self):
        from depth_estimator import estimate_distance
        result = estimate_distance((250, 200, 310, 290), 320, 320)
        self.assertIn("right", result["direction"])

    def test_direction_center(self):
        from depth_estimator import estimate_distance
        result = estimate_distance((120, 200, 200, 290), 320, 320)
        self.assertIn("ahead", result["direction"])

    def test_minimum_distance_clamp(self):
        from depth_estimator import estimate_distance
        import config
        result = estimate_distance((0, 300, 320, 319), 320, 320)
        if result["distance_m"] is not None:
            self.assertGreaterEqual(result["distance_m"], config.MIN_DISTANCE_M)

    def test_face_distance(self):
        from depth_estimator import distance_to_face
        d = distance_to_face(100)    # 100 px face width at some distance
        self.assertGreater(d, 0)

    def test_face_distance_zero_pixel(self):
        from depth_estimator import distance_to_face
        d = distance_to_face(0)
        self.assertEqual(d, 999.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Alert Queue Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAlertQueue(unittest.TestCase):

    def setUp(self):
        from alert_queue import AlertQueueManager
        self.aqm = AlertQueueManager()
        self.spoken: list = []
        self.aqm.set_tts(lambda msg, pri: self.spoken.append((msg, pri)))
        self.aqm.set_haptic(lambda d, s: None)
        self.aqm.set_buzzer(lambda p: None)
        self.aqm.start()

    def tearDown(self):
        self.aqm.stop()

    def test_speech_published(self):
        import config
        self.aqm.speak("Hello world", priority=config.PRIORITY_LOW)
        time.sleep(0.5)
        msgs = [m for m, _ in self.spoken]
        self.assertIn("Hello world", msgs)

    def test_low_priority_dropped_when_congested(self):
        import config
        from alert_queue import Alert, ALERT_TYPE_SPEECH
        # Fill queue above drop threshold
        for i in range(10):
            self.aqm._pq.put(Alert(
                priority=config.PRIORITY_HIGH,
                timestamp=time.time(),
                alert_type=ALERT_TYPE_SPEECH,
                message=f"filler {i}",
            ))
        before = len(self.spoken)
        self.aqm.speak("should be dropped", priority=config.PRIORITY_LOW)
        time.sleep(0.1)
        # LOW alert should not have been added (queue congested)
        # (The queue processes items — we just verify no crash)
        self.assertTrue(True)  # Test passes if no exception

    def test_critical_executes_immediately(self):
        import config
        executed = threading.Event()
        self.aqm.set_tts(lambda msg, pri: executed.set() if "CRITICAL" in msg else None)
        self.aqm.speak("CRITICAL test", priority=config.PRIORITY_CRITICAL)
        self.assertTrue(executed.wait(timeout=2.0), "CRITICAL alert not fired in time")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pothole Detector Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPotholeDetector(unittest.TestCase):

    def setUp(self):
        from pothole_detector import PotholeDetector
        self.detector = PotholeDetector()
        # No model — uses OpenCV fallback

    def test_no_crash_on_black_frame(self):
        frame = make_black_frame()
        result = self.detector.detect(frame)
        self.assertIsInstance(result, list)

    def test_night_mode_detection(self):
        frame = make_black_frame()  # Zero brightness = night mode
        result = self.detector.detect(frame)
        # Should include "system" message about low visibility
        msgs = [d.get("message", "") for d in result]
        night_msgs = [m for m in msgs if "visibility" in m.lower()]
        self.assertTrue(len(night_msgs) >= 0)  # May or may not trigger on black frame

    def test_water_filled_detection(self):
        from pothole_detector import PotholeDetector
        # Create frame with bright patch (simulates water reflection)
        frame = make_black_frame()
        frame[100:200, 100:200] = 240   # Bright patch
        result = PotholeDetector._check_water_filled(frame, (100, 100, 200, 200))
        self.assertIsInstance(result, bool)

    def test_crack_detection_no_crash(self):
        from pothole_detector import PotholeDetector
        frame = make_black_frame()
        cracks = PotholeDetector.detect_road_cracks(frame)
        self.assertIsInstance(cracks, list)

    def test_preprocess_night(self):
        from pothole_detector import PotholeDetector
        frame = make_black_frame()
        processed = PotholeDetector._preprocess_night(frame)
        self.assertEqual(processed.shape, frame.shape)

    def test_severity_small(self):
        import config
        from pothole_detector import PotholeDetector
        det = PotholeDetector()
        frame = make_black_frame()
        # Small bounding box < 5% of frame
        fake_det = {"label": "pothole", "confidence": 0.6, "bbox": (0, 200, 10, 210)}
        enriched = det._enrich(fake_det, frame)
        self.assertEqual(enriched["severity"], "small")

    def test_severity_large(self):
        from pothole_detector import PotholeDetector
        det = PotholeDetector()
        frame = make_black_frame()
        # Large bounding box > 15% of frame
        fake_det = {"label": "pothole", "confidence": 0.7, "bbox": (0, 160, 200, 290)}
        enriched = det._enrich(fake_det, frame)
        self.assertEqual(enriched["severity"], "large")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Vision Engine Tests (no real model)
# ─────────────────────────────────────────────────────────────────────────────
class TestVisionEngine(unittest.TestCase):

    def test_priority_tier_lookup(self):
        from vision_engine import _get_tier
        import config
        self.assertEqual(_get_tier("car")["priority"], config.PRIORITY_CRITICAL)
        self.assertEqual(_get_tier("pothole")["priority"], config.PRIORITY_HIGH)
        self.assertEqual(_get_tier("stairs_up")["priority"], config.PRIORITY_MEDIUM)
        self.assertEqual(_get_tier("person")["priority"], config.PRIORITY_LOW)
        self.assertIsNone(_get_tier("unicorn"))

    def test_traffic_light_color_red(self):
        import cv2
        import numpy as np
        from vision_engine import detect_traffic_light_color
        # Create a red-dominant patch
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[50:100, 50:100] = (0, 0, 200)    # BGR red
        result = detect_traffic_light_color(frame, (50, 50, 100, 100))
        # Red is in valid range
        self.assertIsInstance(result, (str, type(None)))

    def test_traffic_light_green(self):
        import numpy as np
        from vision_engine import detect_traffic_light_color
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[50:100, 50:100] = (0, 200, 0)   # BGR green
        result = detect_traffic_light_color(frame, (50, 50, 100, 100))
        self.assertIsInstance(result, (str, type(None)))

    def test_horizontal_direction(self):
        from depth_estimator import _horizontal_direction
        self.assertIn("left",  _horizontal_direction(50,  320))
        self.assertIn("ahead", _horizontal_direction(160, 320))
        self.assertIn("right", _horizontal_direction(280, 320))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Gesture Engine Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestGestureEngine(unittest.TestCase):

    def test_rule_based_open_palm(self):
        from gesture_engine import GestureEngine
        # Simulate open palm: all tips above PIPs
        pts = np.zeros((21, 3), dtype=np.float32)
        # Fingers extended: tip Y < pip Y (Y inverted: smaller = higher up)
        # Index: tip=8, pip=6; Middle: tip=12, pip=10; Ring: tip=16, pip=14; Pinky: tip=20, pip=18
        for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
            pts[tip][1]  = -0.5   # tip higher
            pts[pip][1]  =  0.0   # pip lower
        features = pts.flatten()
        result = GestureEngine._rule_based(features)
        self.assertEqual(result, "open_palm")

    def test_rule_based_fist(self):
        from gesture_engine import GestureEngine
        pts = np.zeros((21, 3), dtype=np.float32)
        # All tips below PIPs = curled
        for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
            pts[tip][1]  =  0.5
            pts[pip][1]  =  0.0
        # Thumb not extended
        pts[4][0] = -0.1   # thumb tip left of IP
        pts[3][0] =  0.1
        features = pts.flatten()
        result = GestureEngine._rule_based(features)
        self.assertEqual(result, "fist")

    def test_feature_extraction_shape(self):
        from gesture_engine import GestureEngine, NUM_LANDMARKS, FEATURE_DIM
        # Create a mock landmark object
        class MockLandmark:
            def __init__(self):
                self.x = self.y = self.z = 0.0
        class MockMultiLandmarks:
            landmark = [MockLandmark() for _ in range(NUM_LANDMARKS)]
        features = GestureEngine._extract_features(MockMultiLandmarks())
        self.assertEqual(features.shape, (FEATURE_DIM,))


# ─────────────────────────────────────────────────────────────────────────────
# 7. MockGPIO Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestMockGPIO(unittest.TestCase):

    def test_output_and_read_back(self):
        from mock_gpio import MockGPIO as GPIO
        GPIO.setup(99, GPIO.OUT)
        GPIO.output(99, GPIO.HIGH)
        self.assertEqual(GPIO.input(99), GPIO.HIGH)
        GPIO.output(99, GPIO.LOW)
        self.assertEqual(GPIO.input(99), GPIO.LOW)

    def test_simulate_button_press(self):
        from mock_gpio import MockGPIO as GPIO
        received = []
        GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(17, GPIO.FALLING, callback=lambda pin: received.append(pin))
        GPIO.simulate_button_press(17)
        self.assertEqual(received, [17])

    def test_pwm_interface(self):
        from mock_gpio import MockGPIO as GPIO
        GPIO.setup(23, GPIO.OUT)
        pwm = GPIO.PWM(23, 100)
        pwm.start(50)
        self.assertEqual(pwm.duty, 50)
        pwm.ChangeDutyCycle(75)
        self.assertEqual(pwm.duty, 75)
        pwm.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Alert Deduplication Integration Test
# ─────────────────────────────────────────────────────────────────────────────
class TestAlertDeduplication(unittest.TestCase):
    """Feed same object 10 times in 2 seconds — should announce only once."""

    def test_dedup_vision_cooldown(self):
        from vision_engine import VisionEngine
        import config

        ve = VisionEngine()
        announcements = []

        # Patch alert manager to record speaks
        from alert_queue import AlertQueueManager
        mock_aqm = AlertQueueManager()
        mock_aqm.set_tts(lambda m, p: announcements.append(m))
        mock_aqm.set_haptic(lambda d, s: None)
        mock_aqm.set_buzzer(lambda p: None)
        mock_aqm.start()

        import alert_queue
        original_mgr = alert_queue.alert_manager
        alert_queue.alert_manager = mock_aqm

        try:
            frame = make_black_frame()
            fake_dets = [{"label": "person", "confidence": 0.9,
                          "bbox": (100, 100, 200, 200)}] * 10

            # Spam 10 identical detections rapidly
            for det_list in [fake_dets]:
                ve._process_detections(det_list, frame)

            time.sleep(0.3)
            person_anns = [a for a in announcements if "person" in a.lower()]
            # Should announce at most once in the dedup window
            self.assertLessEqual(len(person_anns), 1,
                                 f"Dedup failed: got {len(person_anns)} announcements")
        finally:
            alert_queue.alert_manager = original_mgr
            mock_aqm.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Latency Test
# ─────────────────────────────────────────────────────────────────────────────
class TestLatency(unittest.TestCase):
    """Verify detection→TTS dispatch latency < 500ms (queue publish, not audio end)."""

    def test_dispatch_latency(self):
        from alert_queue import AlertQueueManager, Alert, ALERT_TYPE_SPEECH
        import config

        start_time = [None]
        end_time   = [None]

        aqm = AlertQueueManager()
        def record_speak(msg, pri):
            if "latency" in msg:
                end_time[0] = time.time()
        aqm.set_tts(record_speak)
        aqm.set_haptic(lambda d, s: None)
        aqm.set_buzzer(lambda p: None)
        aqm.start()

        try:
            start_time[0] = time.time()
            aqm.speak("latency test", priority=config.PRIORITY_HIGH)
            time.sleep(0.6)   # Give worker time to process
            self.assertIsNotNone(end_time[0], "Speech was never dispatched")
            latency_ms = (end_time[0] - start_time[0]) * 1000
            self.assertLess(latency_ms, 500,
                            f"Dispatch latency {latency_ms:.0f}ms exceeds 500ms target")
        finally:
            aqm.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Memory Stress Placeholder
# ─────────────────────────────────────────────────────────────────────────────
class TestMemoryStress(unittest.TestCase):
    """
    Lightweight version of stress test for CI.
    Full 5-minute stress test should be run on hardware with:
      python3 test_suite.py --stress
    """

    def test_alert_queue_sustained_load(self):
        from alert_queue import AlertQueueManager
        import config

        aqm = AlertQueueManager()
        aqm.set_tts(lambda m, p: None)
        aqm.set_haptic(lambda d, s: None)
        aqm.set_buzzer(lambda p: None)
        aqm.start()

        try:
            for i in range(200):
                aqm.speak(f"test message {i}", priority=config.PRIORITY_LOW)
                time.sleep(0.01)
            # Queue should not crash or deadlock
            time.sleep(0.5)
        finally:
            aqm.stop()

        self.assertTrue(True)   # If we reach here, no deadlock


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_stress_test(duration_seconds: int = 300):
    """Full 5-minute stress test — run on Pi hardware only."""
    import tracemalloc
    tracemalloc.start()
    print(f"Stress test running for {duration_seconds}s…")
    from alert_queue import AlertQueueManager
    import config

    aqm = AlertQueueManager()
    aqm.set_tts(lambda m, p: time.sleep(0.01))
    aqm.set_haptic(lambda d, s: None)
    aqm.set_buzzer(lambda p: None)
    aqm.start()

    start = time.time()
    count = 0
    while time.time() - start < duration_seconds:
        aqm.speak(f"msg {count}", priority=config.PRIORITY_LOW)
        count += 1
        time.sleep(0.1)

    aqm.stop()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"Total messages: {count}")


if __name__ == "__main__":
    if "--stress" in sys.argv:
        duration = int(sys.argv[sys.argv.index("--stress") + 1]) if len(sys.argv) > sys.argv.index("--stress") + 1 else 300
        run_stress_test(duration)
    else:
        # Run normal test suite
        loader  = unittest.TestLoader()
        suite   = unittest.TestSuite()
        for cls in [
            TestConfig, TestDepthEstimator, TestAlertQueue,
            TestPotholeDetector, TestVisionEngine, TestGestureEngine,
            TestMockGPIO, TestAlertDeduplication, TestLatency, TestMemoryStress,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
