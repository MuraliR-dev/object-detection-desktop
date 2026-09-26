"""
gui/theme.py
------------
Dark AI/CV-themed stylesheet and colour constants for the entire app.
All colours, fonts and geometry are defined here — nowhere else.
"""

# ─── Colour tokens ────────────────────────────────────────────────────────────
BG_DEEP    = "#0d0f14"   # darkest background
BG_PANEL   = "#12151c"   # side panels / cards
BG_CARD    = "#1a1e2a"   # inner cards / inputs
BG_HOVER   = "#1f2435"   # hover state
ACCENT     = "#00d4ff"   # primary cyan accent
ACCENT_2   = "#7c3aed"   # secondary purple
ACCENT_OK  = "#10b981"   # green — running
ACCENT_WARN= "#f59e0b"   # amber — warning
ACCENT_ERR = "#ef4444"   # red — error / stop
TEXT_PRI   = "#e8eaf0"   # primary text
TEXT_SEC   = "#8b95a8"   # muted / secondary text (improved contrast)
TEXT_DIM   = "#4b5563"   # very dim / disabled
BORDER     = "#1f2d3d"   # card border
BORDER_LIT = "#00d4ff44" # glowing border (accent, semi-transparent)
SLIDER_GRV = "#1f2435"   # slider groove
SCROLLBAR  = "#1f2435"

# ─── Geometry ─────────────────────────────────────────────────────────────────
RADIUS      = "8px"
RADIUS_SM   = "4px"
RADIUS_LG   = "12px"
BTN_H       = "32px"     # reduced from 38px to save vertical space
INPUT_H     = "30px"
PANEL_W_PX  = 320        # widened from 300px for better readability

# ─── Font ─────────────────────────────────────────────────────────────────────
FONT_FAMILY = "Segoe UI"
FONT_MONO   = "Consolas"

# ─── Full QSS stylesheet ──────────────────────────────────────────────────────
QSS = f"""
/* ── Global ──────────────────────────────────────────────────────────── */
* {{
    font-family: "{FONT_FAMILY}", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {TEXT_PRI};
    outline: none;
}}
QMainWindow, QWidget#centralWidget {{
    background: {BG_DEEP};
}}

/* ── Generic QWidget panel ───────────────────────────────────────────── */
QWidget#sidePanel {{
    background: {BG_PANEL};
    border-left: 1px solid {BORDER};
}}
QWidget#previewPanel {{
    background: {BG_DEEP};
}}

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    margin-top: 14px;
    padding: 7px 6px 6px 6px;
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_SEC};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    color: {ACCENT};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
}}

/* ── Primary action buttons ──────────────────────────────────────────── */
QPushButton {{
    background: {BG_HOVER};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    padding: 4px 12px;
    min-height: {BTN_H};
    color: {TEXT_PRI};
    font-weight: 500;
    font-size: 12px;
}}
QPushButton:hover {{
    background: #252a3a;
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton:pressed {{
    background: #1a1e2a;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    border-color: {TEXT_DIM};
    background: {BG_CARD};
}}

/* Accent (primary) button */
QPushButton#btnPrimary {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_2}, stop:1 {ACCENT});
    border: none;
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
}}
QPushButton#btnPrimary:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #9f5afc, stop:1 #22e6ff);
    color: #ffffff;
}}
QPushButton#btnPrimary:pressed  {{ opacity: 0.85; }}
QPushButton#btnPrimary:disabled {{
    background: {BG_HOVER};
    color: {TEXT_DIM};
}}

/* Danger / Stop button */
QPushButton#btnDanger {{
    background: transparent;
    border: 1px solid {ACCENT_ERR};
    color: {ACCENT_ERR};
    font-weight: 600;
}}
QPushButton#btnDanger:hover {{
    background: {ACCENT_ERR};
    color: #ffffff;
}}

/* Success / positive button */
QPushButton#btnSuccess {{
    background: transparent;
    border: 1px solid {ACCENT_OK};
    color: {ACCENT_OK};
    font-weight: 600;
}}
QPushButton#btnSuccess:hover {{
    background: {ACCENT_OK};
    color: #ffffff;
}}

/* ── Labels ──────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {TEXT_PRI};
}}
QLabel#labelTitle {{
    font-size: 20px;
    font-weight: 800;
    color: {ACCENT};
    letter-spacing: 1px;
}}
QLabel#labelSubtitle {{
    font-size: 11px;
    color: {TEXT_SEC};
    letter-spacing: 0.5px;
}}
QLabel#statValue {{
    font-family: "{FONT_MONO}";
    font-size: 20px;
    font-weight: 700;
    color: {ACCENT};
}}
QLabel#statLabel {{
    font-size: 10px;
    font-weight: 600;
    color: {TEXT_SEC};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QLabel#statusBadge {{
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: {RADIUS_SM};
}}

/* ── Preview label (the big camera/image canvas) ─────────────────────── */
QLabel#previewLabel {{
    background: #070910;
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    color: {TEXT_SEC};
    font-size: 14px;
}}

/* ── Slider (confidence threshold) ──────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {SLIDER_GRV};
    height: 6px;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_2}, stop:1 {ACCENT});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: 2px solid {BG_DEEP};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: #22e6ff;
}}

/* ── Progress bar ────────────────────────────────────────────────────── */
QProgressBar {{
    background: {SLIDER_GRV};
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 10px;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_2}, stop:1 {ACCENT});
    border-radius: 4px;
}}

/* ── List widget (detections log) ────────────────────────────────────── */
QListWidget {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    font-family: "{FONT_MONO}";
    font-size: 12px;
    alternate-background-color: {BG_CARD};
}}
QListWidget::item {{
    padding: 4px 6px;
    border-bottom: 1px solid {BG_CARD};
}}
QListWidget::item:selected {{
    background: {BG_HOVER};
    color: {ACCENT};
}}

/* ── Scroll bars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_DEEP};
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {SCROLLBAR};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* ── Combo box ───────────────────────────────────────────────────────── */
QComboBox {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    padding: 4px 10px;
    min-height: {INPUT_H};
    color: {TEXT_PRI};
}}
QComboBox:hover  {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    selection-background-color: {BG_HOVER};
}}

/* ── Splitter handle ─────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {BORDER};
}}

/* ── Tool tip ────────────────────────────────────────────────────────── */
QToolTip {{
    background: {BG_CARD};
    border: 1px solid {ACCENT};
    color: {TEXT_PRI};
    padding: 4px 8px;
    border-radius: {RADIUS_SM};
}}
"""
