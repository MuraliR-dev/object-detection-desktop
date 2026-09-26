"""
detector/core.py
----------------
ObjectDetector – the central detection engine.

Design goals
------------
* Load the YOLO model ONCE and reuse it for all detection modes.
* Expose three clean public methods: detect_image, detect_webcam, detect_video.
* Keep drawing / overlay logic in utils.py (separation of concerns).
* Handle all I/O errors gracefully with informative messages.
* Track real FPS using a sliding-window counter.
* Count objects per class per frame / per image.
"""

import os
import time
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_IMG_SIZE,
    DEFAULT_MAX_DET, DEFAULT_CAM_INDEX, DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT,
    DEFAULT_CAM_FPS, MODEL_PATH, OUTPUT_DIR, VIDEO_DISPLAY_WAIT_MS,
)
from .utils import (
    draw_detections, draw_hud, FPSCounter,
    resize_frame, create_output_dir, make_video_writer,
)
from .tracker import TemporalTracker


# ─── Helper: find the most recently modified best.pt in the runs tree ──────────

def find_latest_best_pt(base_dir: str = None) -> Optional[str]:
    """
    Scan runs/detect/ (and runs/custom_training/) for all best.pt files
    and return the path of the most recently modified one.

    Returns None if no best.pt is found.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    search_roots = [
        os.path.join(base_dir, "runs", "detect"),
        os.path.join(base_dir, "runs", "custom_training"),
    ]

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "best.pt" in filenames:
                full = os.path.join(dirpath, "best.pt")
                try:
                    mtime = os.path.getmtime(full)
                    candidates.append((mtime, full))
                except OSError:
                    pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


class ObjectDetector:
    """
    Modular real-time object detector powered by YOLO11n.

    Parameters
    ----------
    model_path : str
        Path to the .pt weights file.  Passed to load_model() but the
        auto-discover logic in load_model() will still pick up the latest
        custom best.pt alongside the default pretrained model.
    conf_threshold : float
        Minimum confidence score (0-1) to keep a detection.
    iou_threshold : float
        NMS IoU threshold.
    img_size : int
        Inference resolution fed to YOLO (pixels).
    device : str
        Torch device string, e.g. 'cpu', 'cuda:0', ''.  Empty = auto.
    verbose : bool
        Whether YOLO prints per-inference logs.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = "",
        verbose: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.img_size       = img_size
        self.verbose        = verbose

        self._fps_counter = FPSCounter()
        self.total_detections: int = 0
        self.tracker = TemporalTracker(confirmation_frames=3, max_missed_frames=2)
        self.load_model(model_path, device)

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self, model_path: str = MODEL_PATH, device: str = ""):
        """
        Load / reload models.

        Always loads the default pretrained model (models/yolo11n.pt).
        If a custom best.pt is provided via model_path OR auto-discovered
        in the runs/ tree, it is loaded as the custom model.
        """
        from ultralytics import YOLO
        from .config import BASE_DIR

        default_model_path = os.path.join(BASE_DIR, "models", "yolo11n.pt")

        # Determine custom model path:
        # 1. If caller explicitly passed a path that is NOT the default model → use it
        # 2. Otherwise, auto-discover the latest best.pt
        norm_passed = os.path.normcase(os.path.abspath(model_path)) if model_path else ""
        norm_default = os.path.normcase(os.path.abspath(default_model_path))
        is_custom_path = (
            model_path
            and os.path.isfile(model_path)
            and norm_passed != norm_default
        )

        if is_custom_path:
            custom_model_path = model_path
        else:
            # Auto-discover latest best.pt from training runs
            custom_model_path = find_latest_best_pt(BASE_DIR)

        self._default_model = None
        self._custom_model = None
        self._device = device

        # --- Load default (pretrained) model ---
        if os.path.isfile(default_model_path):
            print(f"[Detector] Default model: {default_model_path}")
            self._default_model = YOLO(default_model_path)
            if device:
                self._default_model.to(device)
        else:
            print(f"[Detector] WARNING: Default model not found at {default_model_path}")

        # --- Load custom model ---
        if custom_model_path and os.path.isfile(custom_model_path):
            print(f"[Detector] Custom model:  {custom_model_path}")
            self._custom_model = YOLO(custom_model_path)
            if device:
                self._custom_model.to(device)
            print(f"[Detector] Custom classes: {self._custom_model.names}")
        else:
            print("[Detector] No custom model found — using default model only.")

        if not self._default_model and not self._custom_model:
            raise FileNotFoundError("No models could be loaded.")

        # model_path attribute always reflects the default model
        self.model_path = default_model_path

        # names / num_classes reflects the default model (COCO 80 classes)
        # plus custom classes shown separately
        ref = self._default_model or self._custom_model
        self.names = ref.names
        self.num_classes = len(self.names)

        print(f"[Detector] Models ready | task={ref.task}")

    def reload_custom_model(self, custom_pt_path: str):
        """
        Hot-swap the custom model without reloading the default model.
        Call this immediately after training completes.
        """
        from ultralytics import YOLO

        if not os.path.isfile(custom_pt_path):
            raise FileNotFoundError(f"best.pt not found: {custom_pt_path}")

        print(f"[Detector] Reloading custom model: {custom_pt_path}")
        self._custom_model = YOLO(custom_pt_path)
        if self._device:
            self._custom_model.to(self._device)
        print(f"[Detector] Custom classes: {self._custom_model.names}")

    @property
    def custom_model_names(self) -> dict:
        """Return custom model class names, or empty dict if no custom model."""
        if self._custom_model is not None:
            return dict(self._custom_model.names)
        return {}

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray) -> List[dict]:
        """
        Run YOLO on a single frame using all loaded models.

        Returns
        -------
        list[dict]
            Each dict has keys:
                class_id    (int)   – unique id (custom classes offset to avoid collisions)
                class_name  (str)
                confidence  (float)
                x1, y1, x2, y2  (int)  – bounding box pixels
                source_model (str)  – "default" or "custom"
        """
        detections = []

        # ── Default model ───────────────────────────────────────────────────
        if self._default_model is not None:
            res = self._default_model(
                frame,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=DEFAULT_MAX_DET,
                verbose=self.verbose,
            )
            boxes = res[0].boxes
            names = self._default_model.names
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_threshold:
                        continue
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "class_id":    cls_id,
                        "class_name":  names.get(cls_id, f"cls{cls_id}"),
                        "confidence":  conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "source_model": "default",
                    })

        # ── Custom model ────────────────────────────────────────────────────
        if self._custom_model is not None:
            res = self._custom_model(
                frame,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=DEFAULT_MAX_DET,
                verbose=self.verbose,
            )
            boxes = res[0].boxes
            names = self._custom_model.names
            # Offset custom class IDs so they never collide with default IDs (0-79)
            id_offset = 10000
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_threshold:
                        continue
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = names.get(cls_id, f"custom{cls_id}")
                    detections.append({
                        "class_id":    id_offset + cls_id,
                        "class_name":  class_name,
                        "confidence":  conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "source_model": "custom",
                    })
                    # Log custom detections (but not every frame — only when found)
                    # We log here once per detection; callers can suppress by verbose=False
                    if self.verbose:
                        print(f"[Custom] class={class_name} confidence={conf:.2f}")

        # ── Cross-model NMS: suppress duplicates from different models ────
        detections = self._cross_model_nms(detections)
        
        # ── Temporal Tracking ───────────────────────────────────────────────
        detections = self.tracker.update(detections)
        
        return detections

    @staticmethod
    def _cross_model_nms(detections: List[dict], iou_thresh: float = 0.5) -> List[dict]:
        """
        Remove near-duplicate boxes across models.

        If two detections from *different* models overlap heavily (IoU >= thresh)
        AND share the same class_name, keep the one with higher confidence.
        Detections from the same model are never suppressed against each other
        (YOLO already does NMS internally).
        Different class names are NEVER suppressed against each other so that
        'person' and 'atm_card' can coexist in the same region.
        """
        if len(detections) <= 1:
            return detections

        keep = [True] * len(detections)
        for i in range(len(detections)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue
                # Only suppress across different models with same class name
                if detections[i]["source_model"] == detections[j]["source_model"]:
                    continue
                if detections[i]["class_name"] != detections[j]["class_name"]:
                    continue
                iou = ObjectDetector._box_iou(detections[i], detections[j])
                if iou >= iou_thresh:
                    if detections[i]["confidence"] >= detections[j]["confidence"]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        return [d for d, k in zip(detections, keep) if k]

    @staticmethod
    def _box_iou(a: dict, b: dict) -> float:
        """Compute IoU between two detection dicts."""
        x1 = max(a["x1"], b["x1"])
        y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"])
        y2 = min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


    # 2. WEBCAM DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_webcam(
        self,
        cam_index: int = DEFAULT_CAM_INDEX,
        width: int = DEFAULT_CAM_WIDTH,
        height: int = DEFAULT_CAM_HEIGHT,
        save_output: bool = False,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, Dict[str, int], float], None, None]:
        """
        Run real-time detection from a webcam.

        Yields (annotated_frame, counts, fps) for each captured frame.
        The caller is responsible for displaying / using the frames.
        """
        print("[Webcam] Trying DirectShow backend...")
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("[Webcam] Camera opened successfully (DirectShow)")
        else:
            print("[Webcam] DirectShow failed, trying default backend...")
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                print("[Webcam] Unable to open camera. Please check camera permissions/device.")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          DEFAULT_CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Webcam] Default detector ready")
        if self._custom_model is not None:
            print(f"[Webcam] Custom detector ready  classes={self._custom_model.names}")
        print(f"[Detector] Webcam {cam_index} opened  |  {actual_w}x{actual_h}")

        writer: Optional[cv2.VideoWriter] = None
        if save_output:
            if output_path is None:
                create_output_dir(OUTPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, "webcam_detection.mp4")
            writer = make_video_writer(output_path, DEFAULT_CAM_FPS,
                                       (actual_w, actual_h))
            print(f"[Detector] Saving webcam output → {output_path}")

        self._fps_counter.reset()
        self.total_detections = 0
        frame_idx = 0

        try:
            fail_count = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count > 10:
                        print("[Detector] Webcam read failed too many times, stopping.")
                        break
                    time.sleep(0.05)
                    continue
                else:
                    fail_count = 0

                results = self._infer(frame)
                annotated, counts = draw_detections(
                    frame.copy(), results, self.conf_threshold
                )

                frame_idx += 1
                fps = self._fps_counter.tick()
                self.total_detections += sum(counts.values())

                annotated = draw_hud(
                    annotated, fps, self.total_detections, counts,
                    self.conf_threshold, f"CAM {cam_index}"
                )

                if writer:
                    writer.write(annotated)

                yield annotated, counts, fps

                if max_frames is not None and frame_idx >= max_frames:
                    break

                if cv2.waitKey(VIDEO_DISPLAY_WAIT_MS) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("[Detector] Webcam detection interrupted by user.")

        finally:
            cap.release()
            if writer:
                writer.release()
            print(f"[Detector] Webcam closed  |  "
                  f"Frames: {frame_idx}  |  "
                  f"Total detections: {self.total_detections}")

    # ──────────────────────────────────────────────────────────────────────────
    # Property: conf_threshold (allows runtime changes)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def conf_threshold(self) -> float:
        return self._conf_threshold

    @conf_threshold.setter
    def conf_threshold(self, value: float):
        if not 0.0 < value < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1), got {value}")
        self._conf_threshold = value

    # ──────────────────────────────────────────────────────────────────────────
    # Repr
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        custom_info = ""
        if self._custom_model is not None:
            custom_info = f", custom_classes={list(self._custom_model.names.values())}"
        return (
            f"ObjectDetector("
            f"default='{os.path.basename(self.model_path)}', "
            f"conf={self.conf_threshold:.0%}"
            f"{custom_info})"
        )
