"""
run.py
------
Canonical entry point for Object Detection Studio.

Usage
-----
    python run.py                        # default settings
    python run.py --conf 0.5            # custom confidence threshold
    python run.py --cam 1               # non-default webcam index
    python run.py --model models/yolo11n.pt

This file is a thin wrapper that delegates to main.py so that
both `python run.py` and `python main.py` work identically.
"""

import sys
import os

# Ensure the project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Delegate to main entry point
from main import main

if __name__ == "__main__":
    main()
