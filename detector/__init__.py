"""
detector/__init__.py
--------------------
Object Detection Engine Package
Exports the main ObjectDetector class.
"""

from .core import ObjectDetector
from .utils import draw_detections, resize_frame, create_output_dir

__all__ = ["ObjectDetector", "draw_detections", "resize_frame", "create_output_dir"]
__version__ = "1.0.0"
