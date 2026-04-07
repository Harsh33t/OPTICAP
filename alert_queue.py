"""
alert_queue.py — OPTICap Priority Alert Queue Manager
Team Overclocked Minds | B.Tech CSE 2026

Coordinates TTS, haptic, and buzzer output. Prevents sensory overload.
Priority: 1=CRITICAL (bypass), 2=HIGH, 3=MEDIUM, 4=LOW
"""

import queue
import threading
import time
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Callable
import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(config.LOG_DIR, exist_ok=True)

alert_logger = logging.getLogger("opticap.alerts")
alert_logger.setLevel(logging.INFO)

_fh = logging.FileHandler(config.ALERT_LOG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
alert_logger.addHandler(_fh)

# ─────────────────────────────────────────────────────────────────────────────
# Alert Data Model
# ─────────────────────────────────────────────────────────────────────────────
ALERT_TYPE_SPEECH = "SPEECH"
ALERT_TYPE_HAPTIC = "HAPTIC"
ALERT_TYPE_BUZZER = "BUZZER"
ALERT_TYPE_COMBO  = "COMBO"


@dataclass(order=True)
class Alert:
    priority:    int
    timestamp:   float = field(compare=False)
    alert_type:  str   = field(compare=False)
    message:     str   = field(compare=False, default="")
    haptic_data: Optional[dict] = field(compare=False, default=None)
    buzzer_pattern: Optional[str] = field(compare=False, default=None)
    output_device: str = field(compare=False, default="all")
    source_module: str = field(compare=False, default="unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Alert Queue Manager
# ─────────────────────────────────────────────────────────────────────────────
class AlertQueueManager:
    """
    Single manager that accepts all alerts from all modules.
    Routes each to the correct output handler (TTS, haptic, buzzer).
    CRITICAL alerts execute immediately, bypassing the queue.
    """

    MAX_QUEUE_DEPTH   = 10
    LOW_DROP_THRESHOLD = 3   # Drop LOW priority if queue >= this depth

    def __init__(self):
        self._pq: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.MAX_QUEUE_DEPTH)
        self._speech_lock = threading.Lock()
        self._tts_fn:    Optional[Callable[[str, int], None]] = None
        self._haptic_fn: Optional[Callable[[dict, str], None]] = None
        self._buzzer_fn: Optional[Callable[[str], None]] = None
        self._running    = False
        self._worker_thread: Optional[threading.Thread] = None
        self._last_speech_finish = 0.0

    # ── Dependency Injection ───────────────────────────────────────────────
    def set_tts(self, fn: Callable[[str, int], None]):
        self._tts_fn = fn

    def set_haptic(self, fn: Callable[[dict, str], None]):
        self._haptic_fn = fn

    def set_buzzer(self, fn: Callable[[str], None]):
        self._buzzer_fn = fn

    # ── Public API ─────────────────────────────────────────────────────────
    def publish(self, alert: Alert):
        """Submit an alert. CRITICAL alerts bypass the queue immediately."""
        alert_logger.info(
            f"[{alert.source_module}] PRI={alert.priority} TYPE={alert.alert_type} "
            f"MSG={alert.message!r}")

        if alert.priority == config.PRIORITY_CRITICAL:
            self._execute_immediate(alert)
            return

        # Drop LOW alerts if queue is congested
        if alert.priority == config.PRIORITY_LOW:
            if self._pq.qsize() >= self.LOW_DROP_THRESHOLD:
                alert_logger.debug(f"Dropped LOW alert (queue depth {self._pq.qsize()}): {alert.message!r}")
                return

        try:
            self._pq.put_nowait(alert)
        except queue.Full:
            alert_logger.warning(f"Alert queue full. Dropping: {alert.message!r}")

    def speak(self, message: str, priority: int = config.PRIORITY_LOW,
              source: str = "unknown"):
        """Convenience helper: create and publish a SPEECH alert."""
        self.publish(Alert(
            priority=priority,
            timestamp=time.time(),
            alert_type=ALERT_TYPE_SPEECH,
            message=message,
            source_module=source,
        ))

    def haptic(self, haptic_data: dict, side: str = "both",
               priority: int = config.PRIORITY_MEDIUM, source: str = "unknown"):
        """Convenience helper: create a HAPTIC alert."""
        self.publish(Alert(
            priority=priority,
            timestamp=time.time(),
            alert_type=ALERT_TYPE_HAPTIC,
            haptic_data=haptic_data,
            output_device=side,
            source_module=source,
        ))

    def buzzer(self, pattern: str, priority: int = config.PRIORITY_HIGH,
               source: str = "unknown"):
        """Convenience helper: create a BUZZER alert."""
        self.publish(Alert(
            priority=priority,
            timestamp=time.time(),
            alert_type=ALERT_TYPE_BUZZER,
            buzzer_pattern=pattern,
            source_module=source,
        ))

    def combo(self, message: str, haptic_data: dict,
              buzzer_pattern: Optional[str], side: str,
              priority: int = config.PRIORITY_HIGH, source: str = "unknown"):
        """Convenience helper: speech + haptic + optional buzzer together."""
        self.publish(Alert(
            priority=priority,
            timestamp=time.time(),
            alert_type=ALERT_TYPE_COMBO,
            message=message,
            haptic_data=haptic_data,
            buzzer_pattern=buzzer_pattern,
            output_device=side,
            source_module=source,
        ))

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, name="AlertQueueWorker", daemon=True)
        self._worker_thread.start()
        alert_logger.info("AlertQueueManager started.")

    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        alert_logger.info("AlertQueueManager stopped.")

    # ── Internal ───────────────────────────────────────────────────────────
    def _worker(self):
        while self._running:
            try:
                alert = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            self._execute(alert)
            self._pq.task_done()

    def _execute_immediate(self, alert: Alert):
        """Execute in a new thread to avoid blocking caller."""
        t = threading.Thread(target=self._execute, args=(alert,), daemon=True)
        t.start()

    def _execute(self, alert: Alert):
        if alert.alert_type == ALERT_TYPE_SPEECH:
            self._do_speech(alert.message, alert.priority)

        elif alert.alert_type == ALERT_TYPE_HAPTIC:
            self._do_haptic(alert.haptic_data, alert.output_device)

        elif alert.alert_type == ALERT_TYPE_BUZZER:
            self._do_buzzer(alert.buzzer_pattern)

        elif alert.alert_type == ALERT_TYPE_COMBO:
            # TTS and haptic can run simultaneously on separate hardware
            if alert.buzzer_pattern:
                bt = threading.Thread(
                    target=self._do_buzzer, args=(alert.buzzer_pattern,), daemon=True)
                bt.start()
            if alert.haptic_data:
                ht = threading.Thread(
                    target=self._do_haptic,
                    args=(alert.haptic_data, alert.output_device), daemon=True)
                ht.start()
            if alert.message:
                self._do_speech(alert.message, alert.priority)

    def _do_speech(self, message: str, priority: int):
        if not message or not self._tts_fn:
            return
        with self._speech_lock:
            self._tts_fn(message, priority)

    def _do_haptic(self, data: dict, side: str):
        if not data or not self._haptic_fn:
            return
        self._haptic_fn(data, side)

    def _do_buzzer(self, pattern: Optional[str]):
        if not pattern or not self._buzzer_fn:
            return
        self._buzzer_fn(pattern)


# ── Singleton ─────────────────────────────────────────────────────────────────
alert_manager = AlertQueueManager()
