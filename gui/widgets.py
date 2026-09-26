"""
gui/widgets.py
--------------
Custom reusable widgets used by the main window.

StatCard      – animated numeric stat card (FPS, count, conf)
DetectionLog  – scrollable list of latest detections
StatusBar     – bottom status strip
PreviewLabel  – image/video canvas that scales content to fit
"""

from PySide6.QtCore  import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui   import QPixmap, QImage, QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QListWidget, QListWidgetItem, QSizePolicy,
    QFrame, QGraphicsOpacityEffect,
)

from .theme import ACCENT, ACCENT_OK, ACCENT_ERR, ACCENT_WARN, TEXT_SEC, BG_CARD, BG_DEEP, FONT_MONO


# ══════════════════════════════════════════════════════════════════════════════
# PreviewLabel – fills available space and scales pixmap keeping aspect ratio
# ══════════════════════════════════════════════════════════════════════════════

class PreviewLabel(QLabel):
    """Central canvas that displays detected frames scaled to widget size."""

    def __init__(self, placeholder: str = "No input selected", parent=None):
        super().__init__(parent)
        self.setObjectName("previewLabel")
        self.setAlignment(Qt.AlignCenter)
        self.setText(placeholder)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 360)
        self._raw_pixmap: QPixmap | None = None

    def set_pixmap(self, pixmap: QPixmap):
        self._raw_pixmap = pixmap
        self._scale_to_fit()

    def resizeEvent(self, event):
        self._scale_to_fit()
        super().resizeEvent(event)

    def _scale_to_fit(self):
        if self._raw_pixmap is None:
            return
        scaled = self._raw_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def clear_preview(self, msg: str = ""):
        self._raw_pixmap = None
        self.clear()
        self.setText(msg or "No input selected")


# ══════════════════════════════════════════════════════════════════════════════
# StatCard – animated numeric display
# ══════════════════════════════════════════════════════════════════════════════

class StatCard(QWidget):
    """Shows a title + large numeric value with a subtle colour accent."""

    def __init__(self, title: str, unit: str = "", colour: str = ACCENT, parent=None):
        super().__init__(parent)
        self._colour = colour
        self._unit   = unit

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)

        self._lbl_title = QLabel(title.upper())
        self._lbl_title.setObjectName("statLabel")
        self._lbl_title.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; font-weight:700; letter-spacing:1px;")

        self._lbl_value = QLabel("–")
        self._lbl_value.setObjectName("statValue")
        self._lbl_value.setStyleSheet(
            f"font-family:'{FONT_MONO}'; font-size:20px; font-weight:800; color:{colour};"
        )
        self._lbl_value.setAlignment(Qt.AlignLeft)

        layout.addWidget(self._lbl_title)
        layout.addWidget(self._lbl_value)

        self.setStyleSheet(
            f"background:{BG_CARD}; border:1px solid #1f2d3d; border-radius:8px;"
        )
        self.setMinimumHeight(58)

    def set_value(self, value, decimals: int = 0):
        if isinstance(value, float):
            text = f"{value:.{decimals}f}"
        else:
            text = str(value)
        if self._unit:
            text += f" {self._unit}"
        self._lbl_value.setText(text)

    def set_colour(self, colour: str):
        self._lbl_value.setStyleSheet(
            f"font-family:'{FONT_MONO}'; font-size:20px; font-weight:800; color:{colour};"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DetectionLog – scrolling detection list
# ══════════════════════════════════════════════════════════════════════════════

class DetectionLog(QListWidget):
    """Auto-scrolling list that shows the last N detection events."""

    MAX_ROWS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QListWidget.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def log_frame(self, counts: dict, fps: float):
        """Add one entry per detected class in this frame."""
        if not counts:
            return
        for cls, n in sorted(counts.items()):
            item = QListWidgetItem(f"  {cls:<18} x{n:<4}  {fps:5.1f} fps")
            item.setForeground(QColor(ACCENT))
            self.addItem(item)
            # Prune old rows
            while self.count() > self.MAX_ROWS:
                self.takeItem(0)
        self.scrollToBottom()

    def clear_log(self):
        self.clear()


# ══════════════════════════════════════════════════════════════════════════════
# StatusBadge – coloured pill label for camera/mode state
# ══════════════════════════════════════════════════════════════════════════════

class StatusBadge(QLabel):
    """A rounded pill that shows IDLE / RUNNING / ERROR with a colour flash."""

    STATES = {
        "idle":    (TEXT_SEC,    BG_CARD,      "⬤  IDLE"),
        "running": (ACCENT_OK,   "#0d2818",    "⬤  RUNNING"),
        "paused":  (ACCENT_WARN, "#1f1708",    "⬤  PAUSED"),
        "error":   (ACCENT_ERR,  "#2a0d0d",    "⬤  ERROR"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusBadge")
        self.setAlignment(Qt.AlignCenter)
        self.set_state("idle")

    def set_state(self, state: str, extra: str = ""):
        colour, bg, text = self.STATES.get(state, self.STATES["idle"])
        label = text + (f"  {extra}" if extra else "")
        self.setText(label)
        self.setStyleSheet(
            f"color:{colour}; background:{bg}; border:1px solid {colour}44;"
            f"border-radius:4px; padding:3px 10px; font-size:11px; font-weight:700;"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Divider
# ══════════════════════════════════════════════════════════════════════════════

class HDivider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setStyleSheet("color: #1f2d3d;")
        self.setFixedHeight(1)
