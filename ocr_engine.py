"""
ocr_engine.py — OPTICap Visual OCR Engine
Team Overclocked Minds | B.Tech CSE 2026

Reads text from the camera frame using Llama 3.2 Vision via GitHub Models (Azure AI).
"""

import time
import base64
import cv2
import threading
import logging
from openai import OpenAI
import config

logger = logging.getLogger("opticap.ocr")

SYSTEM_PROMPT = """You are the OCR engine inside OPTICap, a wearable assistive cap for blind users in India. Your output is read aloud via text-to-speech.

Rules:
- Read ALL visible text in natural reading order, top to bottom
- Speak numbers naturally: "50 rupees", "5 milligrams", "Gate 3"
- If text is Hindi or Hinglish, read it and give the English meaning immediately after without announcing you are translating
- Add ONE short context sentence at the end if helpful
- Never start with "I can see", "I notice", or "The image shows"
- No bullet points or markdown — plain flowing sentences only
- Keep response under 60 words
- If no text visible, say exactly: "No readable text in view."
- If text is blurred or cut off, read what is visible and say "rest is unclear"
"""

class OCREngine:
    def __init__(self):
        self.client = None
        self.model = "meta/Llama-3.2-11B-Vision-Instruct"
        self.last_call_time = 0.0
        self.cooldown = 3.0

    def setup(self):
        token = getattr(config, "GITHUB_TOKEN", None)
        if not token or token == "your_token_here":
            logger.warning("GITHUB_TOKEN not configured in config.py")
            return
        
        try:
            self.client = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=token
            )
            logger.info("OCR Engine connected to GitHub Models.")
        except Exception as e:
            logger.error(f"Failed to init OCR Engine: {e}")

    def read_frame_async(self, frame, speak_callback):
        """Spawns background thread for OCR to avoid blocking main loop."""
        if not self.client:
            speak_callback("OCR strictly needs a GitHub token configured.")
            return

        now = time.time()
        if now - self.last_call_time < self.cooldown:
            return  # Cooldown active
            
        self.last_call_time = now
        
        speak_callback("Reading text.")

        def _worker():
            try:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64_img = base64.b64encode(buf.tobytes()).decode('utf-8')

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read all text visible in this image for a blind person."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]
                        }
                    ],
                    max_tokens=120,
                    temperature=0.1
                )
                raw_text = resp.choices[0].message.content.strip()
                
                # Strip markdown for TTS
                clean_text = raw_text.replace("*", "").replace("#", "")
                speak_callback(clean_text)
                
            except Exception as e:
                logger.error(f"OCR API Error: {e}")
                speak_callback("Error connecting to OCR engine.")

        threading.Thread(target=_worker, daemon=True).start()

ocr_engine = OCREngine()
