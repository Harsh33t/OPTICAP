"""
tts_engine.py — OPTICap Text-to-Speech Engine (Piper, offline)
Team Overclocked Minds | B.Tech CSE 2026

Priority queue for speech:
  Level 1 (CRITICAL) → interrupt current speech immediately
  Level 2 (HIGH)     → queue after current utterance
  Level 3 (LOW)      → drop if queue has 2+ pending items
"""

import queue
import threading
import subprocess
import tempfile
import os
import time
import logging
import config

logger = logging.getLogger("opticap.tts")


class TTSEngine:
    """
    Wraps Piper TTS for offline, low-latency, non-blocking speech output.
    Exposes: speak(text, priority)
    """

    QUEUE_MAX = 8

    def __init__(self):
        self._q: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.QUEUE_MAX)
        self._current_proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._running = False
        self._worker: threading.Thread | None = None
        self._speaking = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._worker = threading.Thread(
            target=self._run, name="TTSWorker", daemon=True)
        self._worker.start()
        logger.info("TTSEngine started.")

    def stop(self):
        self._running = False
        self._kill_current()
        if self._worker:
            self._worker.join(timeout=2.0)
        logger.info("TTSEngine stopped.")

    # ── Public API ────────────────────────────────────────────────────────
    def speak(self, text: str, priority: int = config.PRIORITY_LOW):
        """
        Speak `text` at the given priority level.
        Priority 1 = interrupt everything immediately.
        """
        if not text:
            return

        if priority == config.PRIORITY_CRITICAL:
            # Kill any ongoing speech, prepend to front of queue
            self._kill_current()
            self._enqueue(text, priority, force=True)
            return

        if priority == config.PRIORITY_LOW:
            if self._q.qsize() >= 2:
                logger.debug(f"TTS LOW dropped (queue={self._q.qsize()}): {text!r}")
                return

        self._enqueue(text, priority)

    # ── Internal ──────────────────────────────────────────────────────────
    def _enqueue(self, text: str, priority: int, force: bool = False):
        entry = (priority, time.time(), text)
        try:
            if force:
                # To force to front: clear queue and re-add after our entry
                pending = []
                while not self._q.empty():
                    try:
                        pending.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                self._q.put_nowait(entry)
                for item in pending:
                    try:
                        self._q.put_nowait(item)
                    except queue.Full:
                        break
            else:
                self._q.put_nowait(entry)
        except queue.Full:
            logger.warning(f"TTS queue full, dropping: {text!r}")

    def _run(self):
        while self._running:
            try:
                priority, ts, text = self._q.get(timeout=0.3)
            except queue.Empty:
                continue

            self._speak_blocking(text)
            self._q.task_done()

    def _speak_blocking(self, text: str):
        """Call Piper and pipe output to aplay (raw PCM → 3.5mm jack)."""
        logger.info(f"TTS speaking: {text!r}")
        self._speaking.set()
        try:
            # Piper writes 22050 Hz 16-bit mono PCM to stdout, aplay plays it
            piper_cmd = [
                "piper",
                "--model", self._voice_path(),
                "--output-raw",
                "--length-scale", str(1 / config.TTS_SPEED),
            ]
            aplay_cmd = ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1", "-"]

            with self._proc_lock:
                piper_proc = subprocess.Popen(
                    piper_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                aplay_proc = subprocess.Popen(
                    aplay_cmd,
                    stdin=piper_proc.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                piper_proc.stdout.close()  # Allow aplay to receive EOF
                self._current_proc = aplay_proc

            # Send text to Piper
            piper_proc.communicate(input=text.encode("utf-8"))
            aplay_proc.wait()

        except FileNotFoundError:
            logger.error("Piper TTS not installed. Install with: pip install piper-tts")
            self._fallback_espeak(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self._speaking.clear()
            with self._proc_lock:
                self._current_proc = None

    def _fallback_espeak(self, text: str):
        """Fallback to espeak if Piper is unavailable."""
        try:
            subprocess.run(
                ["espeak", "-s", "160", text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10
            )
        except Exception as e:
            logger.error(f"espeak fallback failed: {e}")

    def _kill_current(self):
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
                self._current_proc = None

    @staticmethod
    def _voice_path() -> str:
        """Return path to Piper .onnx model file."""
        voice = config.TTS_VOICE
        # Default search path for Piper voices
        default_dirs = [
            "/usr/share/piper-voices",
            os.path.expanduser("~/.local/share/piper"),
            "/home/opticap/models/piper",
        ]
        for d in default_dirs:
            candidate = os.path.join(d, f"{voice}.onnx")
            if os.path.isfile(candidate):
                return candidate
        # Fallback: return name and let Piper resolve
        return voice


# ── Singleton ─────────────────────────────────────────────────────────────────
tts_engine = TTSEngine()
