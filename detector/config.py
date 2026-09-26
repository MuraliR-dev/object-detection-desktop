"""
detector/config.py
------------------
Central configuration constants for the Object Detection Engine.
All tunable parameters live here so the rest of the code stays clean.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# ── Detection defaults ────────────────────────────────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.40   # 40 % minimum confidence to show a detection
DEFAULT_IOU_THRESHOLD  = 0.45   # NMS IoU threshold
DEFAULT_IMG_SIZE       = 640    # YOLO inference resolution (px)
DEFAULT_MAX_DET        = 300    # maximum detections per frame

# ── Webcam defaults ───────────────────────────────────────────────────────────
DEFAULT_CAM_INDEX = 0           # first webcam
DEFAULT_CAM_WIDTH  = 1280
DEFAULT_CAM_HEIGHT = 720
DEFAULT_CAM_FPS    = 30

# ── Video processing ─────────────────────────────────────────────────────────
VIDEO_DISPLAY_WAIT_MS = 1       # cv2.waitKey delay during video playback

# ── Drawing / overlay ────────────────────────────────────────────────────────
BOX_THICKNESS        = 2
FONT                 = 0        # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE           = 0.55
FONT_THICKNESS       = 1
LABEL_PADDING        = 4        # px padding inside label background
OVERLAY_ALPHA        = 0.35     # transparency for the HUD overlay panel

# Colour palette (BGR) – one per COCO class; cycles if more classes exist
COLOUR_PALETTE = [
    (56,  56,  255), (151,  57, 255), (31,  112, 255), (29, 178, 255),
    (49,  210, 207), (10,  249, 133), (68,  212,  52), (147, 212,  21),
    (255, 182,   0), (255, 127,   0), (220,  75,   0), (255,  25,   0),
    (100,   0, 192), (0,    150, 255), (0,   200, 150), (80,  200,  80),
]

# ── FPS smoothing ─────────────────────────────────────────────────────────────
FPS_SMOOTHING_WINDOW = 30       # number of recent frames used for FPS average
