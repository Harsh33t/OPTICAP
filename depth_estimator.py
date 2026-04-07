"""
depth_estimator.py — OPTICap Monocular Depth & Step-Count Estimator
Team Overclocked Minds | B.Tech CSE 2026

Estimates ground-plane distance from bounding box using camera geometry.
No stereo cameras or LIDAR required.

Formula: D = H × tan(θ)
  H = camera height (config.CAMERA_HEIGHT_M)
  θ = vertical angle from horizontal to object, corrected for camera tilt
"""

import math
import logging
import config

logger = logging.getLogger("opticap.depth")


def estimate_distance(
    bbox: tuple[int, int, int, int],
    frame_width:  int = config.CAMERA_WIDTH,
    frame_height: int = config.CAMERA_HEIGHT,
) -> dict:
    """
    Estimate distance from camera to an object on the ground plane.

    Args:
        bbox: (x1, y1, x2, y2) pixel coordinates of the bounding box.
        frame_width:  width of the camera frame in pixels.
        frame_height: height of the camera frame in pixels.

    Returns:
        dict with keys:
          distance_m  – estimated ground distance in metres (float)
          steps       – estimated step count to reach the object (int)
          direction   – horizontal direction string
          overhead    – True if object appears elevated (no ground formula)
    """
    x1, y1, x2, y2 = bbox

    # ── Horizontal direction from bounding box centre ────────────────────
    cx = (x1 + x2) / 2.0
    direction = _horizontal_direction(cx, frame_width)

    # ── Overhead check: bottom edge above frame midpoint ─────────────────
    if y2 < (frame_height / 2):
        logger.debug("Object appears overhead — no ground-plane formula.")
        return {
            "distance_m": None,
            "steps": None,
            "direction": direction,
            "overhead": True,
        }

    # ── Normalised bottom-edge Y ──────────────────────────────────────────
    y_bottom_norm = y2 / frame_height   # 0 = top, 1 = bottom of frame

    # ── Vertical angle calculation ────────────────────────────────────────
    # Map y_bottom_norm (0→1) to an angle relative to the camera optical axis.
    # The frame spans config.CAMERA_VFOV_DEG vertically.
    # y=0   → top of frame    → +VFOV/2 from optical axis
    # y=1   → bottom of frame → -VFOV/2 from optical axis
    # Camera is tilted downward by CAMERA_TILT_DEG.
    # θ is measured from horizontal upward = positive above horizon.

    vfov_half = config.CAMERA_VFOV_DEG / 2.0          # degrees
    angle_from_axis = (0.5 - y_bottom_norm) * config.CAMERA_VFOV_DEG  # degrees from optical axis
    # Angle from horizontal (positive = above horizon, negative = below)
    angle_from_horiz = angle_from_axis + config.CAMERA_TILT_DEG

    # Convert to radians
    theta_rad = math.radians(angle_from_horiz)

    # ── Distance formula ──────────────────────────────────────────────────
    # D = H / tan(depression_angle)
    # When theta_from_horiz is negative (camera angled below horizon to object):
    # depression angle = -theta_from_horiz
    # D = H / tan(depression_angle)
    depression_rad = -theta_rad   # positive when camera looks down to object

    if depression_rad <= 0:
        # Object is at or above horizon — cannot use ground plane formula
        return {
            "distance_m": None,
            "steps": None,
            "direction": direction,
            "overhead": True,
        }

    raw_distance = config.CAMERA_HEIGHT_M / math.tan(depression_rad)
    distance_m = max(raw_distance, config.MIN_DISTANCE_M)

    steps = int(math.ceil(distance_m / config.STRIDE_LENGTH_M))

    result = {
        "distance_m": round(distance_m, 2),
        "steps": steps,
        "direction": direction,
        "overhead": False,
    }
    logger.debug(f"Depth estimate: {result}")
    return result


def _horizontal_direction(cx: float, frame_width: int) -> str:
    """Return 'on your left' / 'directly ahead' / 'on your right'."""
    third = frame_width / 3.0
    if cx < third:
        return "on your left"
    elif cx > 2 * third:
        return "on your right"
    else:
        return "directly ahead"


def distance_to_face(pixel_width: int) -> float:
    """
    Estimate face distance using known average face width.
    D = (known_width_px × focal_length) / pixel_width
    known_width_px derived from KNOWN_FACE_WIDTH_CM / cm-per-pixel at 1m.

    Returns distance in centimetres.
    """
    if pixel_width <= 0:
        return 999.0
    d_cm = (config.KNOWN_FACE_WIDTH_CM * config.FOCAL_LENGTH_PX) / pixel_width
    return round(d_cm, 1)


def describe_distance(distance_m: float | None) -> str:
    """Human-readable distance string."""
    if distance_m is None:
        return "unknown distance"
    if distance_m < 1.0:
        return "very close"
    if distance_m < 3.0:
        return f"{distance_m:.1f} metres away"
    return f"about {int(round(distance_m))} metres away"
