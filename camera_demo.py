"""
camera_demo.py — OPTICap Live Camera Demo
Team Overclocked Minds | B.Tech CSE 2026

Dual-engine detection:
  1. YOLOv8n   → bounding boxes (every frame)
  2. Vision AI → natural-language scene understanding (every 7 sec to preserve free quotas)

Supported free providers:
    python camera_demo.py --provider gemini --key YOUR_GEMINI_KEY
    python camera_demo.py --provider together --key YOUR_TOGETHER_KEY
    python camera_demo.py --provider github --key YOUR_GITHUB_PAT
"""

import cv2
import numpy as np
import time
import sys
import math
import argparse
import threading
import queue
import textwrap
import base64
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics"); sys.exit(1)

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

CONFIDENCE        = 0.35       
DEDUP_ALERT_SEC   = 3.0        
CAMERA_HEIGHT_M   = 1.6
CAMERA_TILT_DEG   = 15
CAMERA_VFOV_DEG   = 41.4
STRIDE_M          = 0.65
MIN_DIST_M        = 0.30
AI_INTERVAL       = 7.0   # 7 SECONDS = guarantees we never hit Gemini 15 req/minute limit!

TIER1 = {"car","truck","motorcycle","bus","bicycle","person walking"}
TIER2 = {"pothole","manhole","fire hydrant","suitcase","backpack on ground"}
TIER3 = {"stairs","ramp","curb","stop sign"}
TIER4 = {"person","dog","cat","bird","traffic light","bench","chair",
         "bottle","cup","laptop","cell phone","umbrella"}

C_RED    = (50,  50,  230)
C_ORANGE = (40,  140, 255)
C_YELLOW = (50,  220, 240)
C_GREEN  = (60,  210, 100)
C_WHITE  = (240, 240, 240)
C_DARK   = (18,  18,  28)
C_PANEL  = (25,  25,  40)

TIER_COLOR = {1: C_RED, 2: C_ORANGE, 3: C_YELLOW, 4: C_GREEN}
TIER_LABEL = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}

AI_PROMPT = (
    "You are the voice of OPTICap, an AI assistant for blind users wearing a smart cap. "
    "Analyze this scene and in 2 SHORT sentences describe: "
    "(1) the most important hazard or object the user must know about, "
    "(2) a safe navigation instruction. "
    "Be direct, no filler words. Example: 'A car is parked on the left sidewalk. "
    "The path ahead is clear, proceed straight.'"
)

OCR_PROMPT = """You are the OCR engine inside OPTICap, a wearable assistive cap for blind users in India. Your output is read aloud via text-to-speech.

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

PROVIDERS = {
    "together": {"base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", "display": "Together AI"},
    "github": {"base_url": "https://models.inference.ai.azure.com", "model": "Llama-3.2-11B-Vision-Instruct", "display": "GitHub"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "display": "OpenAI"}
}

def get_tier(label):
    l = label.lower()
    if l in TIER1: return 1
    if l in TIER2: return 2
    if l in TIER3: return 3
    if l in TIER4: return 4
    return None

def direction_str(cx, w):
    t = w / 3
    if cx < t: return "LEFT"
    if cx > 2*t: return "RIGHT"
    return "AHEAD"

def estimate_dist(y2, h):
    if y2 < h / 2: return None, None
    y_norm = y2 / h
    ang = (0.5 - y_norm) * CAMERA_VFOV_DEG + CAMERA_TILT_DEG
    dep = -math.radians(ang)
    if dep <= 0: return None, None
    d = max(CAMERA_HEIGHT_M / math.tan(dep), MIN_DIST_M)
    return round(d, 1), max(1, int(math.ceil(d / STRIDE_M)))

def put_text_shaded(img, text, pos, scale=0.55, color=C_WHITE, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def filled_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.75, r=6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(overlay, (cx,cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

# ─────────────────────────────────────────────────────────────────────────────
# Workers
# ─────────────────────────────────────────────────────────────────────────────
class GeminiWorker:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-2.0-flash")
        self.display_name = "Gemini AI"
        self._in_q, self._out_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame, prompt_type="scene"):
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            prompt_str = OCR_PROMPT if prompt_type == "ocr" else AI_PROMPT
            self._in_q.put_nowait({"frame": buf.tobytes(), "type": prompt_type, "prompt": prompt_str})
        except: pass

    def get_result(self):
        try: return self._out_q.get_nowait()
        except: return None

    def _run(self):
        while self._running:
            try:
                task = self._in_q.get(timeout=1.0)
                jpg_bytes = task["frame"]
                prompt = task.get("prompt", AI_PROMPT)
                req_type = task.get("type", "scene")
            except: continue
            try:
                resp = self._model.generate_content(
                    [prompt, {"mime_type": "image/jpeg", "data": base64.b64encode(jpg_bytes).decode()}],
                    safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE}
                )
                res = resp.text.strip().replace("*", "").replace("#", "")
            except Exception as e:
                res = f"[Gemini Limit Reached]" if "429" in str(e) else f"[Gemini Error]"
            try: self._out_q.put_nowait({"type": req_type, "text": res})
            except: pass

    def stop(self): self._running = False


class OpenAIVisionWorker:
    def __init__(self, api_key: str, provider_id: str):
        cfg = PROVIDERS[provider_id]
        self.model, self.display_name = cfg["model"], cfg["display"]
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        self._in_q, self._out_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame, prompt_type="scene"):
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            prompt_str = OCR_PROMPT if prompt_type == "ocr" else AI_PROMPT
            self._in_q.put_nowait({"frame": buf.tobytes(), "type": prompt_type, "prompt": prompt_str})
        except: pass

    def get_result(self):
        try: return self._out_q.get_nowait()
        except: return None

    def _run(self):
        while self._running:
            try:
                task = self._in_q.get(timeout=1.0)
                jpg_bytes = task["frame"]
                prompt = task.get("prompt", AI_PROMPT)
                req_type = task.get("type", "scene")
            except: continue
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": [{"type": "text", "text": "Execute task."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(jpg_bytes).decode()}"}}]}],
                    temperature=0.1
                )
                res = resp.choices[0].message.content.strip().replace("*", "").replace("#", "")
            except Exception as e:
                res = f"[{self.display_name} Limit Reached. Using YOLO.]" if ("429" in str(e) or "quota" in str(e).lower()) else f"[{self.display_name} Error: {e}]"
            try: self._out_q.put_nowait({"type": req_type, "text": res})
            except: pass

    def stop(self): self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# HUD & Main
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, yolo_alerts, ai_text, fps, ai_name, ai_status, ocr_text=None, ocr_status="ready"):
    h, w = frame.shape[:2]
    filled_rounded_rect(frame, 0, 0, w, 52, C_DARK, alpha=0.85, r=0)
    put_text_shaded(frame, "OPTICap  LIVE DEMO", (14, 34), scale=0.85, color=(200,230,255), thickness=2, font=cv2.FONT_HERSHEY_DUPLEX)
    
    ai_col = (100,255,140) if ai_name else (100,100,120)
    disp_name = (ai_name.upper() if getattr(ai_name, "upper", None) else str(ai_name).upper()) if ai_name else "AI"
    put_text_shaded(frame, f"FPS {fps:04.1f}   {disp_name}: {ai_status}   OCR: {ocr_status}", (w-550, 34), scale=0.52, color=ai_col)

    # ── AI Scene description panel ──
    if ai_text:
        lines = textwrap.wrap(ai_text, width=int(w / 8.5))[:4]
        py = 58
        filled_rounded_rect(frame, 0, py, w, py + len(lines)*24 + 18, (10,40,20), alpha=0.82, r=0)
        cv2.line(frame, (0, py), (w, py), (60,200,100), 1)
        cv2.rectangle(frame, (8, py+4), (135, py+22), (40,160,80), -1)
        put_text_shaded(frame, disp_name, (12, py+17), scale=0.42, color=(220,255,220), thickness=1)
        for i, ln in enumerate(lines):
            put_text_shaded(frame, ln, (142, py+18+i*24), scale=0.52, color=(210,255,210), thickness=1)

    # ── AI OCR Panel ──
    if ocr_text:
        lines = textwrap.wrap(ocr_text, width=int(w / 8.5))[:4]
        py = 58 + 120 # Offset below Scene Description
        filled_rounded_rect(frame, 0, py, w, py + len(lines)*24 + 18, (10,30,50), alpha=0.9, r=0)
        cv2.line(frame, (0, py), (w, py), (60,100,200), 1)
        cv2.rectangle(frame, (8, py+4), (135, py+22), (60,100,180), -1)
        put_text_shaded(frame, f"OCR TEXT", (12, py+17), scale=0.42, color=(220,255,255), thickness=1)
        for i, ln in enumerate(lines):
            put_text_shaded(frame, ln, (142, py+18+i*24), scale=0.52, color=(210,230,255), thickness=1)

    if yolo_alerts:
        unique = []
        for tier, text in yolo_alerts:
            if text not in [u[1] for u in unique]: unique.append((tier, text))
        py2 = h - (len(unique[:5]) * 38 + 14) - 4
        filled_rounded_rect(frame, 0, py2, w, h, C_PANEL, alpha=0.88, r=0)
        cv2.line(frame, (0, py2), (w, py2), (80,80,120), 1)
        for i, (tier, txt) in enumerate(unique[:5]):
            y = py2 + 10 + i * 38
            cv2.rectangle(frame, (6,y), (100,y+26), TIER_COLOR[tier], -1)
            put_text_shaded(frame, TIER_LABEL[tier], (10, y+18), scale=0.46, color=(10,10,10), thickness=1)
            max_chars = int((w - 120) / 7.5)
            display_txt = txt if len(txt) <= max_chars else txt[:max_chars-1]+"…"
            put_text_shaded(frame, display_txt, (110, y+19), scale=0.56, color=C_WHITE, thickness=1)


def draw_detections(frame, boxes):
    for (x1,y1,x2,y2, label, conf, tier) in boxes:
        col = TIER_COLOR.get(tier, C_GREEN)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        tag = f"{label}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ty = max(y1-4, th+6)
        cv2.rectangle(frame, (x1,ty-th-5),(x1+tw+8,ty+2), col, -1)
        put_text_shaded(frame, tag, (x1+4, ty-1), scale=0.48, color=(10,10,10), thickness=1)

def run_demo(camera_idx=0, video_path=None, api_key=None, provider="gemini"):
    model = YOLO("yolov8n.pt")
    ai_worker, ai_name = None, None

    if api_key:
        try:
            if provider == "gemini" and _GEMINI_OK:
                ai_worker = GeminiWorker(api_key)
            elif provider in PROVIDERS and _OPENAI_OK:
                ai_worker = OpenAIVisionWorker(api_key, provider)
            
            if ai_worker: ai_name = ai_worker.display_name
        except Exception as e: print(f"✗ Setup failed: {e}")

    cap = cv2.VideoCapture(video_path or camera_idx)
    if not video_path:
        cap.set(3, 800); cap.set(4, 600)

    dedup = {}
    fps_buf = deque(maxlen=25)
    ai_text = ""
    ai_status = "ready"
    last_ai_req = 0.0

    ocr_text = ""
    ocr_status = "ready"

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            if video_path: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break

        h, w = frame.shape[:2]
        now = time.time()
        results = model(frame, imgsz=320, conf=CONFIDENCE, verbose=False)[0]

        boxes_info, yolo_alerts = [], []
        for box in results.boxes:
            label, conf = model.names[int(box.cls[0])], float(box.conf[0])
            tier = get_tier(label)
            if not tier: continue
            
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            boxes_info.append((x1,y1,x2,y2, label, conf, tier))
            
            cx = (x1+x2)/2.0
            dir_s = direction_str(cx, w)
            alert = f"{label} {dir_s}"
            if now - dedup.get((label, dir_s), 0.0) >= DEDUP_ALERT_SEC:
                dedup[(label, dir_s)] = now
                yolo_alerts.append((tier, alert))

        if ai_worker and now - last_ai_req >= AI_INTERVAL:
            ai_worker.submit_frame(cv2.resize(frame, (480, 360)), prompt_type="scene")
            last_ai_req, ai_status = now, "analyzing…"

        if ai_worker:
            res = ai_worker.get_result()
            if res: 
                if res["type"] == "scene":
                    ai_text, ai_status = res["text"], "live"
                elif res["type"] == "ocr":
                    ocr_text, ocr_status = res["text"], "live"

        draw_detections(frame, boxes_info)
        fps_buf.append(time.time() - t0)
        draw_hud(frame, sorted(yolo_alerts, key=lambda x: x[0]), ai_text, 1.0/(sum(fps_buf)/len(fps_buf)) if fps_buf else 0, ai_name, ai_status, ocr_text, ocr_status)

        cv2.imshow("OPTICap Live Demo [Q=quit, O=OCR]", frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), 27):
            break
        elif k == ord('s'):
            cv2.imwrite("opticap_shot.jpg", frame)
            print("  📸 Screenshot saved")
        elif k == ord('o'):
            if ai_worker:
                print("  👁️ Triggering OCR Read...")
                ocr_text = ""
                ocr_status = "reading..."
                ai_worker.submit_frame(cv2.resize(frame, (640, 480)), prompt_type="ocr")
            else:
                print("  ✗ Cannot run OCR without API key. Add --key!")

    cap.release()
    cv2.destroyAllWindows()
    if ai_worker: ai_worker.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", type=str)
    p.add_argument("--key", type=str)
    p.add_argument("--provider", type=str, choices=["gemini", "together", "github", "openai"], default="gemini")
    args = p.parse_args()
    run_demo(args.camera, args.video, args.key, args.provider)
