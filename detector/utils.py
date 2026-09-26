"""
detector/utils.py
-----------------
Pure drawing / helper utilities.
No YOLO / model code here – keeps concerns separated.
"""

import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    BOX_THICKNESS, COLOUR_PALETTE, FONT, FONT_SCALE, FONT_THICKNESS,
    LABEL_PADDING, OVERLAY_ALPHA, FPS_SMOOTHING_WINDOW,
)


# ─── Colour helper ────────────────────────────────────────────────────────────

def get_colour(class_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR colour for a given class id."""
    return COLOUR_PALETTE[class_id % len(COLOUR_PALETTE)]


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: List[dict],
    conf_threshold: float = 0.4,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Draw bounding boxes, class names and confidence percentages on *frame*.

    Parameters
    ----------
    detections : list[dict]
        Each dict has keys: class_id, class_name, confidence,
        x1, y1, x2, y2, source_model.

    Returns
    -------
    annotated_frame : np.ndarray
        Frame with all boxes drawn in-place.
    counts : dict
        {class_name: count} for all detected objects in this frame.
    """
    counts: Dict[str, int] = {}

    if not detections:
        return frame, counts

    for det in detections:
        conf = det["confidence"]
        if conf < conf_threshold:
            continue

        cls_id = det["class_id"]
        label  = det["class_name"]
        colour = get_colour(cls_id)

        # Bounding box
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, BOX_THICKNESS)

        # Label string
        text = f"{label}  {conf:.0%}"

        # Measure text so we can draw a filled background
        (tw, th), baseline = cv2.getTextSize(
            text, FONT, FONT_SCALE, FONT_THICKNESS
        )
        pad = LABEL_PADDING
        # Clamp label above box; if no room above, put it inside
        label_y1 = max(y1 - th - 2 * pad, 0)
        label_y2 = label_y1 + th + 2 * pad
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 2 * pad, label_y2),
                      colour, cv2.FILLED)

        # Text colour: white on dark bg, dark on light bg
        brightness = (colour[0] * 0.114 + colour[1] * 0.587 + colour[2] * 0.299)
        text_colour = (0, 0, 0) if brightness > 160 else (255, 255, 255)
        cv2.putText(frame, text, (x1 + pad, label_y2 - pad - baseline),
                    FONT, FONT_SCALE, text_colour, FONT_THICKNESS,
                    cv2.LINE_AA)

        # Update per-frame counts
        counts[label] = counts.get(label, 0) + 1

    return frame, counts


def draw_hud(
    frame: np.ndarray,
    fps: float,
    total_detections: int,
    counts: Dict[str, int],
    conf_threshold: float,
    source_label: str = "",
) -> np.ndarray:
    """
    Overlay a semi-transparent HUD panel (top-left) showing:
      • FPS, total detections so far, per-class counts, confidence threshold.

    Uses a blended rectangle so the video beneath stays visible.
    """
    lines = [
        f"FPS: {fps:5.1f}",
        f"Conf >= {conf_threshold:.0%}",
        f"Detected: {total_detections}",
    ]
    if counts:
        lines.append("─" * 18)
        for name, n in sorted(counts.items()):
            lines.append(f"  {name}: {n}")
    if source_label:
        lines.append(f"[{source_label}]")

    line_h   = 20
    panel_w  = 180
    panel_h  = len(lines) * line_h + 10
    panel_h  = min(panel_h, frame.shape[0])

    # Draw blended panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    for i, line in enumerate(lines):
        y = 16 + i * line_h
        if y > frame.shape[0] - 4:
            break
        cv2.putText(frame, line, (8, y),
                    FONT, 0.48, (200, 255, 200), 1, cv2.LINE_AA)

    return frame


# ─── FPS calculator ───────────────────────────────────────────────────────────

class FPSCounter:
    """Sliding-window real FPS calculator."""

    def __init__(self, window: int = FPS_SMOOTHING_WINDOW):
        self._window = window
        self._times: deque = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Call once per processed frame. Returns smoothed FPS."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)

    def reset(self):
        self._times.clear()
        self._last = time.perf_counter()


# ─── Frame helpers ────────────────────────────────────────────────────────────

def resize_frame(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> np.ndarray:
    """Downscale frame while preserving aspect ratio (never upscales)."""
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─── Output dir helper ────────────────────────────────────────────────────────

def create_output_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path


# ─── Video writer helper ──────────────────────────────────────────────────────

def make_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],   # (width, height)
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    """Return a configured VideoWriter, creating parent dirs as needed."""
    create_output_dir(os.path.dirname(output_path) or ".")
    cc = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(output_path, cc, fps, frame_size)
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for: {output_path}")
    return writer
