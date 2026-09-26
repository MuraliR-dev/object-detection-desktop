"""
gui/workers.py
--------------
QThread worker classes for non-blocking YOLO inference.

Three workers:
  WebcamWorker  – continuously captures + infers from a webcam
  VideoWorker   – processes a video file frame-by-frame
  ImageWorker   – runs single-image inference on a thread

All workers emit signals back to the main thread so the GUI never blocks.
"""

import os
import time
from typing import Dict, Optional

import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker


# ─── Helper: convert BGR numpy frame → QImage bytes ──────────────────────────
# We pass raw bytes + dimensions; the main thread converts to QPixmap.
# This avoids importing QtGui in a worker thread.

def _frame_to_bytes(frame: np.ndarray):
    """Return (bytes, h, w, bytes_per_line) for a BGR frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return rgb.tobytes(), h, w, ch * w


# ══════════════════════════════════════════════════════════════════════════════
# Webcam Worker
# ══════════════════════════════════════════════════════════════════════════════

class WebcamWorker(QThread):
    """
    Captures frames from a webcam, runs YOLO inference, and emits results.

    Signals
    -------
    frame_ready   – (bytes, h, w, bpl, counts_dict, fps)
    error         – (str)  error message
    status        – (str)  informational status message
    finished      – ()     emitted when the loop exits cleanly
    """

    frame_ready = Signal(bytes, int, int, int, dict, float)
    error       = Signal(str)
    status      = Signal(str)
    finished    = Signal()

    def __init__(self, detector, cam_index: int = 0, parent=None):
        super().__init__(parent)
        self._detector   = detector
        self._cam_index  = cam_index
        self._mutex      = QMutex()
        self._stop_flag  = False
        self._conf       = detector.conf_threshold

    # ── Public control API (called from main thread) ───────────────────────

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_flag = True

    def set_conf(self, value: float):
        with QMutexLocker(self._mutex):
            self._conf = value

    def _should_stop(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stop_flag

    # ── Thread entry point ─────────────────────────────────────────────────

    def run(self):
        # Try DirectShow first (Windows); fall back to default backend
        cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status.emit("DirectShow failed, trying default backend...")
            cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            self.error.emit(
                f"Cannot open webcam (index {self._cam_index}).\n"
                "Check that the camera is connected and not used by another app."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status.emit(f"Webcam {self._cam_index} opened  {w}x{h}")

        # Log which models are active
        print("[Webcam] ============================================")
        print(f"[Webcam] Default detector ready: {self._detector.model_path}")
        if getattr(self._detector, '_custom_model', None) is not None:
            print(f"[Webcam] Custom detector ready")
            print(f"[Webcam] Custom model path:   {getattr(self._detector._custom_model, 'ckpt_path', 'unknown')}")
            print(f"[Webcam] Custom classes:       {self._detector._custom_model.names}")
        else:
            print("[Webcam] No custom model loaded — only pretrained model active")
        print("[Webcam] ============================================")
        
        # Reset tracker to prevent stale tracks from old sessions
        from detector.tracker import TemporalTracker
        self._detector.tracker = TemporalTracker(confirmation_frames=2, max_missed_frames=3)

        from detector.utils import draw_detections, draw_hud, FPSCounter
        fps_counter = FPSCounter()
        frame_idx   = 0
        consecutive_fails = 0

        try:
            while not self._should_stop():
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails > 10:
                        self.error.emit("Webcam stream lost after 10 consecutive read failures.")
                        break
                    time.sleep(0.02)
                    continue
                consecutive_fails = 0

                # Sync conf threshold
                with QMutexLocker(self._mutex):
                    conf = self._conf
                self._detector.conf_threshold = conf

                # Inference
                results = self._detector._infer(frame)
                annotated, counts = draw_detections(
                    frame.copy(), results, conf
                )
                fps = fps_counter.tick()
                frame_idx += 1

                annotated = draw_hud(
                    annotated, fps, frame_idx, counts, conf,
                    f"CAM {self._cam_index}"
                )

                data, fh, fw, bpl = _frame_to_bytes(annotated)
                self.frame_ready.emit(data, fh, fw, bpl, counts, fps)

        except Exception as e:
            self.error.emit(f"Webcam worker error: {e}")
        finally:
            cap.release()
            self.status.emit("Webcam closed")
            self.finished.emit()


