"""
main.py
-------
Application entry point.

Usage
-----
    python main.py                      # default (conf=0.40)
    python main.py --conf 0.5           # custom confidence
    python main.py --cam 1              # non-default camera
    python main.py --model models/yolo11n.pt
"""

import sys
import os
import argparse

# Ensure project root is on sys.path so `detector` and `gui` are importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Object Detection Studio")
    p.add_argument("--model",  default=os.path.join(ROOT, "models", "yolo11n.pt"),
                   help="Path to YOLO .pt weights")
    p.add_argument("--conf",   type=float, default=0.40,
                   help="Default confidence threshold (0-1)")
    p.add_argument("--device", default="",
                   help="Torch device (cpu / cuda:0 / …)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── PySide6 high-DPI / OpenGL init (before QApplication) ─────────────────
    from PySide6.QtCore    import Qt
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from PySide6.QtGui     import QPixmap, QColor, QFont

    # High-DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Object Detection Studio")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("YOLO11n")

    # ── Splash screen ──────────────────────────────────────────────────────────
    splash_pix = QPixmap(480, 240)
    splash_pix.fill(QColor("#0d0f14"))
    splash = QSplashScreen(splash_pix)
    splash.setFont(QFont("Segoe UI", 11))
    splash.showMessage(
        "  Loading YOLO11n model…  Please wait.",
        Qt.AlignBottom | Qt.AlignLeft,
        QColor("#00d4ff"),
    )
    splash.show()
    app.processEvents()

    # ── Load detector (heavy; done before window shows) ───────────────────────
    try:
        from detector import ObjectDetector
        detector = ObjectDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device,
            verbose=False,
        )
    except Exception as e:
        splash.close()
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(
            None, "Startup Error",
            f"Failed to load the YOLO model:\n\n{e}\n\n"
            "Make sure models/yolo11n.pt exists.\n"
            "Run  python test_detection.py  to download it."
        )
        sys.exit(1)

    # ── Build & show main window ───────────────────────────────────────────────
    from gui.main_window import MainWindow
    window = MainWindow(detector)
    splash.finish(window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
