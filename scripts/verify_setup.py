"""
verify_setup.py
---------------
Verifies that all required packages are correctly installed.
Run this before anything else to confirm your environment is healthy.
"""

import sys
import importlib

REQUIRED = [
    ("cv2",         "opencv-python"),
    ("numpy",       "numpy"),
    ("ultralytics", "ultralytics"),
    ("torch",       "torch"),
    ("PIL",         "pillow"),
]

def check_package(import_name, pip_name):
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK]  {pip_name:<20} version: {version}")
        return True
    except ImportError:
        print(f"  [FAIL] {pip_name:<20} NOT FOUND - run: pip install {pip_name}")
        return False

def main():
    print("=" * 55)
    print("  Object Detection Project - Environment Check")
    print("=" * 55)
    print(f"\nPython: {sys.version}")
    print(f"Executable: {sys.executable}\n")

    print("Checking packages:")
    results = [check_package(imp, pip) for imp, pip in REQUIRED]

    print()
    if all(results):
        print("[SUCCESS] All packages are installed correctly.")
        print("          You can now run: python test_detection.py")
    else:
        print("[WARNING] Some packages are missing. Install them first.")
    print("=" * 55)

if __name__ == "__main__":
    main()
