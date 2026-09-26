"""
gui/main_window.py
------------------
MainWindow – the entire GUI assembled from modular pieces.

Layout
------
┌──────────────────────────────────────┬──────────────────┐
│            Preview Panel             │   Control Panel  │
│   (PreviewLabel fills this area)     │  ─ Source group  │
│                                      │  ─ Stats group   │
│                                      │  ─ Settings group│
│                                      │  ─ Actions group │
│                                      │  ─ Detection log │
└──────────────────────────────────────┴──────────────────┘
│                  Status bar                              │
└──────────────────────────────────────────────────────────┘
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np

from PySide6.QtCore  import Qt, QTimer, Slot, QSize
from PySide6.QtGui   import QPixmap, QImage, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QComboBox, QSplitter,
    QProgressBar, QStatusBar, QSizePolicy, QSpacerItem,
    QFrame, QScrollArea,
)

from .theme   import QSS, ACCENT, ACCENT_OK, ACCENT_ERR, ACCENT_WARN, TEXT_SEC, PANEL_W_PX, BG_DEEP
from .widgets import PreviewLabel, StatCard, DetectionLog, StatusBadge, HDivider
from .workers import WebcamWorker
from .training_dialog import TrainingDialog


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bytes_to_pixmap(data: bytes, h: int, w: int, bpl: int) -> QPixmap:
    """Convert raw RGB bytes to a QPixmap (always on the main thread)."""
    img = QImage(data, w, h, bpl, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


# ══════════════════════════════════════════════════════════════════════════════
# MainWindow
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self, detector):
        super().__init__()
        self._detector = detector
        self._webcam_worker: WebcamWorker | None = None

        self._current_pixmap: QPixmap | None = None
        self._last_frame_bgr: np.ndarray | None = None   # for screenshot
        self._mode = "idle"   # idle | webcam
        self._total_detections = 0
        self._frame_count = 0

        self._setup_window()
        self._build_ui()
        self._connect_signals()
        self._apply_theme()

        # Update status bar every second
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_statusbar)
        self._status_timer.start(1000)

    # ──────────────────────────────────────────────────────────────────────────
    # Window setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle("Object Detection Studio  ·  YOLO11n")
        self.setMinimumSize(900, 600)
        self.resize(1300, 800)
        # Centre on screen
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width()  - self.width())  // 2,
            (screen.height() - self.height()) // 2,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: preview area ─────────────────────────────────────────────
        preview_panel = QWidget()
        preview_panel.setObjectName("previewPanel")
        pv_layout = QVBoxLayout(preview_panel)
        pv_layout.setContentsMargins(12, 10, 6, 8)
        pv_layout.setSpacing(6)

        # Title bar
        title_bar = QHBoxLayout()
        title_bar.setSpacing(8)
        lbl_title = QLabel("OBJECT DETECTION STUDIO")
        lbl_title.setObjectName("labelTitle")
        lbl_sub = QLabel(f"YOLO11n  ·  {self._detector.num_classes} COCO classes")
        lbl_sub.setObjectName("labelSubtitle")
        title_bar.addWidget(lbl_title)
        title_bar.addStretch()
        title_bar.addWidget(lbl_sub)
        pv_layout.addLayout(title_bar)

        # Preview label — expands to fill all available space
        self.preview = PreviewLabel("Select a source to begin detection")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pv_layout.addWidget(self.preview, stretch=1)

        root.addWidget(preview_panel, stretch=1)

        # ── Right: control panel ───────────────────────────────────────────
        # Outer container — fixed width, full height
        side_outer = QWidget()
        side_outer.setObjectName("sidePanel")
        side_outer.setFixedWidth(PANEL_W_PX)
        side_outer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        outer_layout = QVBoxLayout(side_outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Scroll area wraps all sidebar content — prevents clipping when window is short
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{ background: #0d0f14; width: 6px; margin: 0; }}"
            f"QScrollBar::handle:vertical {{ background: #1f2435; border-radius: 3px; min-height: 24px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )

        # Inner widget that scroll area contains
        side_inner = QWidget()
        side_inner.setObjectName("sidePanelInner")
        side_inner.setStyleSheet("background: transparent;")
        side_layout = QVBoxLayout(side_inner)
        side_layout.setContentsMargins(8, 10, 8, 10)
        side_layout.setSpacing(6)

        # ── Status badge row ───────────────────────────────────────────────
        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        badge_lbl = QLabel("STATUS")
        badge_lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; font-weight:700; letter-spacing:1px;")
        self.status_badge = StatusBadge()
        badge_row.addWidget(badge_lbl)
        badge_row.addStretch()
        badge_row.addWidget(self.status_badge)
        side_layout.addLayout(badge_row)

                # -- Model Selection ------------------------------------------------
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(8, 6, 8, 8)
        model_layout.setSpacing(5)
        
        self.lbl_current_model = QLabel(f"Current Model:\n{os.path.basename(self._detector.model_path)}")
        self.lbl_current_model.setStyleSheet("color:#8b95a8; font-size:11px; font-weight:bold;")
        
        self.btn_load_model = QPushButton("  Load Custom .pt Model")
        self.btn_load_model.setToolTip("Select a trained YOLO model")
        self.btn_load_model.setMinimumHeight(32)

        self.btn_train_model = QPushButton("  Add / Train New Object")
        self.btn_train_model.setToolTip("Create a dataset and train a custom YOLO model")
        self.btn_train_model.setMinimumHeight(32)
        
        model_layout.addWidget(self.lbl_current_model)
        model_layout.addWidget(self.btn_load_model)
        model_layout.addWidget(self.btn_train_model)
        side_layout.addWidget(model_group)


        # -- Source group ───────────────────────────────────────────────────
        src_group = QGroupBox("Input Source")
        src_layout = QVBoxLayout(src_group)
        src_layout.setContentsMargins(8, 6, 8, 8)
        src_layout.setSpacing(5)

        # Camera index selector
        cam_row = QHBoxLayout()
        cam_row.setSpacing(6)
        cam_lbl = QLabel("Camera:")
        cam_lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:12px;")
        cam_lbl.setFixedWidth(56)
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["0 — Default", "1", "2", "3"])
        self.cam_combo.setToolTip("Select webcam device index")
        cam_row.addWidget(cam_lbl)
        cam_row.addWidget(self.cam_combo, 1)
        src_layout.addLayout(cam_row)

        self.btn_webcam = QPushButton("▶  Start Webcam")
        self.btn_webcam.setObjectName("btnPrimary")
        self.btn_webcam.setToolTip("Start/stop live webcam detection  (W)")
        self.btn_webcam.setMinimumHeight(32)

        self.btn_stop = QPushButton("  Stop")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop current detection  (Esc)")
        self.btn_stop.setMinimumHeight(32)

        src_layout.addWidget(self.btn_webcam)
        src_layout.addWidget(self.btn_stop)
        side_layout.addWidget(src_group)

        # ── Stats group ────────────────────────────────────────────────────
        stats_group = QGroupBox("Live Statistics")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(6, 6, 6, 6)
        stats_layout.setSpacing(5)
        stats_layout.setColumnStretch(0, 1)
        stats_layout.setColumnStretch(1, 1)

        self.stat_fps    = StatCard("FPS",     "fps", ACCENT)
        self.stat_obj    = StatCard("Objects", "",    "#10b981")
        self.stat_conf   = StatCard("Thresh",  "%",   "#f59e0b")
        self.stat_frames = StatCard("Frames",  "",    "#7c3aed")

        stats_layout.addWidget(self.stat_fps,    0, 0)
        stats_layout.addWidget(self.stat_obj,    0, 1)
        stats_layout.addWidget(self.stat_conf,   1, 0)
        stats_layout.addWidget(self.stat_frames, 1, 1)
        side_layout.addWidget(stats_group)

        # ── Confidence threshold ───────────────────────────────────────────
        conf_group = QGroupBox("Confidence Threshold")
        conf_layout = QVBoxLayout(conf_group)
        conf_layout.setContentsMargins(8, 6, 8, 8)
        conf_layout.setSpacing(4)

        # Value display — full width, centred
        self.conf_label = QLabel("40%")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setStyleSheet(
            f"color:{ACCENT}; font-weight:700; font-size:18px; "
            f"background:transparent; padding:2px 0;"
        )
        conf_layout.addWidget(self.conf_label)

        # Slider row with min/max markers
        slider_row = QHBoxLayout()
        slider_row.setSpacing(4)
        lbl_min = QLabel("5%")
        lbl_min.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px;")
        lbl_max = QLabel("95%")
        lbl_max.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(40)
        self.conf_slider.setTickInterval(5)
        self.conf_slider.setToolTip("Minimum detection confidence (5%–95%)")
        slider_row.addWidget(lbl_min)
        slider_row.addWidget(self.conf_slider, 1)
        slider_row.addWidget(lbl_max)
        conf_layout.addLayout(slider_row)
        side_layout.addWidget(conf_group)

        # ── Actions group ──────────────────────────────────────────────────
        act_group = QGroupBox("Actions")
        act_layout = QVBoxLayout(act_group)
        act_layout.setContentsMargins(8, 6, 8, 8)
        act_layout.setSpacing(5)

        self.btn_screenshot = QPushButton("  Screenshot")
        self.btn_screenshot.setObjectName("btnSuccess")
        self.btn_screenshot.setEnabled(False)
        self.btn_screenshot.setToolTip("Save current frame to outputs/  (S)")
        self.btn_screenshot.setMinimumHeight(32)

        self.btn_clear = QPushButton("  Clear / Reset")
        self.btn_clear.setToolTip("Clear preview and log  (C)")
        self.btn_clear.setMinimumHeight(32)

        self.btn_exit = QPushButton("  Exit")
        self.btn_exit.setObjectName("btnDanger")
        self.btn_exit.setToolTip("Close the application  (Ctrl+Q)")
        self.btn_exit.setMinimumHeight(32)

        act_layout.addWidget(self.btn_screenshot)
        act_layout.addWidget(self.btn_clear)
        act_layout.addWidget(self.btn_exit)
        side_layout.addWidget(act_group)

        # ── Detection log ──────────────────────────────────────────────────
        log_group = QGroupBox("Detection Log")
        log_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 4, 4, 4)
        log_layout.setSpacing(0)
        self.log = DetectionLog()
        self.log.setMinimumHeight(100)
        log_layout.addWidget(self.log)
        side_layout.addWidget(log_group, stretch=1)

        # Put inner widget into scroll area
        scroll.setWidget(side_inner)
        outer_layout.addWidget(scroll)

        root.addWidget(side_outer)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status_lbl = QLabel("Ready")
        self.statusBar().addWidget(self._status_lbl, 1)
        self._time_lbl = QLabel()
        self.statusBar().addPermanentWidget(self._time_lbl)
        self.statusBar().setStyleSheet(
            f"background:#0d0f14; border-top:1px solid #1f2d3d; color:{TEXT_SEC}; font-size:11px;"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.btn_webcam.clicked.connect(self._toggle_webcam)
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_screenshot.clicked.connect(self._take_screenshot)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_load_model.clicked.connect(self._load_custom_model)
        self.btn_train_model.clicked.connect(self._open_train_dialog)
        self.btn_exit.clicked.connect(self.close)

        self.conf_slider.valueChanged.connect(self._on_conf_changed)

        # Keyboard shortcuts
        QShortcut(QKeySequence("W"),        self, self._toggle_webcam)
        QShortcut(QKeySequence("Escape"),   self, self._stop_all)
        QShortcut(QKeySequence("S"),        self, self._take_screenshot)
        QShortcut(QKeySequence("C"),        self, self._clear_all)
        QShortcut(QKeySequence("Ctrl+Q"),   self, self.close)

    def _apply_theme(self):
        self.setStyleSheet(QSS)
        # Initial stat values
        self.stat_fps.set_value(0.0, 1)
        self.stat_obj.set_value(0)
        self.stat_conf.set_value(40)
        self.stat_frames.set_value(0)

    # ──────────────────────────────────────────────────────────────────────────
    # Slots — frame display (called from any worker via signal)
    # ──────────────────────────────────────────────────────────────────────────

    @Slot(bytes, int, int, int, dict, float)
    def _display_frame(self, data: bytes, h: int, w: int, bpl: int,
                       counts: dict, fps: float):
        pixmap = _bytes_to_pixmap(data, h, w, bpl)
        self.preview.set_pixmap(pixmap)
        self._current_pixmap = pixmap

        # Stats
        total = sum(counts.values())
        self._total_detections += total
        self._frame_count += 1

        self.stat_fps.set_value(fps, 1)
        self.stat_obj.set_value(total)
        self.stat_frames.set_value(self._frame_count)

        # Colour FPS green/amber/red
        if fps >= 15:
            self.stat_fps.set_colour(ACCENT_OK)
        elif fps >= 6:
            self.stat_fps.set_colour(ACCENT_WARN)
        else:
            self.stat_fps.set_colour(ACCENT_ERR)

        self.log.log_frame(counts, fps)
        self.btn_screenshot.setEnabled(True)

    # ──────────────────────────────────────────────────────────────────────────
    # Slots — worker lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_worker_status(self, msg: str):
        self._set_status(msg)

    @Slot(str)
    def _on_worker_error(self, msg: str):
        self._set_mode("idle")
        self.status_badge.set_state("error")
        self._set_status(f"Error: {msg}")
        QMessageBox.critical(self, "Detection Error", msg)

    @Slot()
    def _on_worker_finished(self):
        if self._mode not in ("image",):
            self._set_mode("idle")


    # ──────────────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────────────

    # _open_train_dialog and _load_custom_model are defined below (final implementations)


    @Slot()
    def _toggle_webcam(self):
        if self._mode == "webcam":
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        self._stop_all()
        cam_idx = self.cam_combo.currentIndex()
        self._webcam_worker = WebcamWorker(self._detector, cam_index=cam_idx)
        self._webcam_worker.frame_ready.connect(self._display_frame)
        self._webcam_worker.error.connect(self._on_worker_error)
        self._webcam_worker.status.connect(self._on_worker_status)
        self._webcam_worker.finished.connect(self._on_worker_finished)
        self._webcam_worker.set_conf(self.conf_slider.value() / 100.0)
        self._webcam_worker.start()
        self._set_mode("webcam")
        self.btn_webcam.setText("  Stop Webcam")

    def _stop_webcam(self):
        if self._webcam_worker:
            self._webcam_worker.stop()
            self._webcam_worker.wait(3000)
            self._webcam_worker = None
        self.btn_webcam.setText("  Start Webcam")
        self._set_mode("idle")

    @Slot()
    def _stop_all(self):
        if self._webcam_worker:
            self._webcam_worker.stop()
            self._webcam_worker.wait(3000)
            self._webcam_worker = None
        self.btn_webcam.setText("  Start Webcam")
        self._set_mode("idle")

    @Slot()
    def _take_screenshot(self):
        if self._current_pixmap is None:
            return
        from detector.config import OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, f"screenshot_{ts}.jpg")
        self._current_pixmap.save(path, "JPEG", 95)
        self._set_status(f"Screenshot saved → {path}")

    @Slot()
    def _load_custom_model(self):
        """Let user browse to any .pt and hot-swap it as the custom model."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Custom Model", "runs/detect", "YOLO Weights (*.pt)"
        )
        if not path:
            return

        self._stop_all()
        self._set_status(f"Loading custom model {os.path.basename(path)}...")
        try:
            self._detector.reload_custom_model(path)
            custom_names = list(self._detector.custom_model_names.values())
            names_str = ", ".join(custom_names) if custom_names else "none"
            self.lbl_current_model.setText(
                f"Custom Model:\n{os.path.basename(path)}\n"
                f"Classes: {names_str}"
            )
            lbl_sub = self.findChild(QLabel, "labelSubtitle")
            if lbl_sub:
                lbl_sub.setText(f"Default + Custom  *  {len(custom_names)} custom class(es)")
            self._set_status(f"Custom model loaded: {os.path.basename(path)}")
            QMessageBox.information(
                self, "Model Loaded",
                f"Custom model loaded successfully.\n"
                f"Custom classes: {names_str}"
            )
        except Exception as e:
            self._on_worker_error(str(e))

    @Slot()
    def _open_train_dialog(self):
        """Open the training dialog; after training, hot-swap the custom model."""
        dialog = TrainingDialog(self)
        if dialog.exec() and dialog.best_model_path:
            path = dialog.best_model_path
            if not os.path.isfile(path):
                QMessageBox.warning(self, "Model Not Found",
                                    f"best.pt was not found at:\n{path}")
                return
            self._stop_all()
            self._set_status(f"Loading trained model {os.path.basename(path)}...")
            try:
                self._detector.reload_custom_model(path)
                custom_names = list(self._detector.custom_model_names.values())
                names_str = ", ".join(custom_names) if custom_names else "none"
                self.lbl_current_model.setText(
                    f"Custom Model:\n{os.path.basename(path)}\n"
                    f"Classes: {names_str}"
                )
                lbl_sub = self.findChild(QLabel, "labelSubtitle")
                if lbl_sub:
                    lbl_sub.setText(f"Default + Custom  *  {len(custom_names)} custom class(es)")
                self._set_status(f"Custom model active: {names_str}")
                reply = QMessageBox.question(
                    self, "Training Complete",
                    f"Custom model loaded successfully.\n"
                    f"Custom classes: {names_str}\n\n"
                    f"Would you like to test the model on a NEW unseen image now?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self._start_webcam()
            except Exception as e:
                self._on_worker_error(str(e))

    @Slot()
    def _clear_all(self):
        self._stop_all()
        self.preview.clear_preview()
        self.log.clear_log()
        self._current_pixmap = None
        self._total_detections = 0
        self._frame_count = 0
        self.stat_fps.set_value(0.0, 1)
        self.stat_obj.set_value(0)
        self.stat_frames.set_value(0)
        self.btn_screenshot.setEnabled(False)
        self._set_status("Cleared")

    @Slot(int)
    def _on_conf_changed(self, value: int):
        conf = value / 100.0
        self.conf_label.setText(f"{value}%")
        self.stat_conf.set_value(value)
        # Live update to active workers
        if self._webcam_worker:
            self._webcam_worker.set_conf(conf)

    # ──────────────────────────────────────────────────────────────────────────
    # Mode management
    # ──────────────────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._mode = mode
        busy = mode == "webcam"

        self.btn_stop.setEnabled(busy)
        self.btn_webcam.setEnabled(True)

        if mode == "webcam":
            self.status_badge.set_state("running", "WEBCAM")
        else:
            self.status_badge.set_state("idle")

    # ──────────────────────────────────────────────────────────────────────────
    # Status helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_lbl.setText(msg)

    def _refresh_statusbar(self):
        self._time_lbl.setText(datetime.now().strftime("%H:%M:%S"))

    # ──────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_all()
        event.accept()
