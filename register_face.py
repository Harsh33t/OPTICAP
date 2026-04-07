"""
register_face.py — OPTICap Face Registration CLI Tool
Team Overclocked Minds | B.Tech CSE 2026

Usage (caretaker runs on Pi):
  python3 register_face.py

Captures 5 photos, saves to /home/opticap/known_faces/<name>/,
then recomputes the encodings database.
"""

import os
import sys
import time
import pickle
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("opticap.register_face")

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency check
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("OpenCV not installed. Run: pip install opencv-python-headless")

try:
    import face_recognition
except ImportError:
    sys.exit("face_recognition not installed. Run: pip install face_recognition")

import config


def capture_face_images(name: str, num_photos: int = 5) -> list[str]:
    """
    Open camera, capture `num_photos` images with 1-second intervals.
    Returns list of saved file paths.
    """
    save_dir = os.path.join(config.KNOWN_FACES_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    saved = []
    print(f"\nCapturing {num_photos} photos for '{name}'.")
    print("Please look at the camera. Photos will be taken every second.\n")

    for i in range(num_photos):
        time.sleep(1.0)

        for _ in range(5):    # Drain buffer before capture
            cap.read()

        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Failed to capture photo {i + 1}. Skipping.")
            continue

        path = os.path.join(save_dir, f"{i + 1:02d}.jpg")
        cv2.imwrite(path, frame)
        saved.append(path)
        print(f"  ✓ Photo {i + 1}/{num_photos} saved → {path}")

    cap.release()
    return saved


def build_encodings_database() -> dict:
    """
    Walk KNOWN_FACES_DIR, compute face encodings for all images.
    Returns {names: [], encodings: []} dict and saves to FACE_ENCODINGS_PATH.
    """
    names     = []
    encodings = []

    base = config.KNOWN_FACES_DIR
    if not os.path.isdir(base):
        print(f"No known faces directory found at {base}")
        return {"names": names, "encodings": encodings}

    for person_name in sorted(os.listdir(base)):
        person_dir = os.path.join(base, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_file in sorted(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_dir, img_file)
            try:
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if not encs:
                    print(f"  Warning: No face detected in {img_path}")
                    continue
                names.append(person_name)
                encodings.append(encs[0])
                print(f"  ✓ Encoded {img_file} → '{person_name}'")
            except Exception as e:
                print(f"  ✗ Error encoding {img_path}: {e}")

    out_path = config.FACE_ENCODINGS_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)

    print(f"\nDatabase saved: {len(names)} face(s) → {out_path}")
    return {"names": names, "encodings": encodings}


def register():
    print("═══ OPTICap Face Registration Tool ═══\n")
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Name cannot be empty.")
        sys.exit(1)

    try:
        paths = capture_face_images(name, num_photos=5)
        if not paths:
            print("No photos captured. Aborting.")
            sys.exit(1)

        print("\nRebuilding face encodings database…")
        data = build_encodings_database()

        print(f"\nRegistration complete. {len(data['names'])} total face(s) in database.")
        print("Reload OPTICap for changes to take effect.")
    except Exception as e:
        print(f"Registration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    register()
