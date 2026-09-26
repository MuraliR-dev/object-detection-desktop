"""
gui/annotation_dialog.py
------------------------
A simple dialog to allow the user to draw bounding boxes on an image.
"""
import os
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox
)

class AnnotationDialog(QDialog):
    def __init__(self, image_path, classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annotate Image")
        self.image_path = image_path
        self.classes = classes
        self.bboxes = [] # list of (class_id, x_center, y_center, width, height) in normalized coords
        
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Tools
        tool_layout = QHBoxLayout()
        self.class_combo = QComboBox()
        for i, c in enumerate(self.classes):
            self.class_combo.addItem(f"{i}: {c}")
            
        btn_save = QPushButton("Save Annotation")
        btn_save.clicked.connect(self._save_and_close)
        btn_clear = QPushButton("Clear Annotations")
        btn_clear.clicked.connect(self._clear)
        
        tool_layout.addWidget(QLabel("Select Class:"))
        tool_layout.addWidget(self.class_combo)
        tool_layout.addStretch()
        tool_layout.addWidget(btn_clear)
        tool_layout.addWidget(btn_save)
        
        layout.addLayout(tool_layout)
        
        # Canvas
        self.canvas = AnnotationCanvas(self.image_path)
        layout.addWidget(self.canvas, 1)

    def _clear(self):
        self.canvas.clear_boxes()
        
    def _save_and_close(self):
        boxes = self.canvas.get_boxes(self.class_combo.currentIndex())
        if not boxes:
            QMessageBox.warning(self, "Warning", "No bounding boxes drawn!")
            return
            
        # YOLO format saving
        txt_path = os.path.splitext(self.image_path)[0] + ".txt"
        try:
            with open(txt_path, 'w') as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

class AnnotationCanvas(QLabel):
    def __init__(self, image_path):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #0d0f14;")
        self._pixmap = QPixmap(image_path)
        self._boxes = [] # (QRectF) relative to original image size
        self._current_rect = None
        self._start_pos = None
        
        self.setCursor(Qt.CrossCursor)
        
    def clear_boxes(self):
        self._boxes = []
        self._current_rect = None
        self.update()
        
    def get_boxes(self, current_class_id):
        # Convert QRectF to YOLO (x_center, y_center, w, h)
        yolo_boxes = []
        w, h = self._pixmap.width(), self._pixmap.height()
        if w == 0 or h == 0: return []
        
        for r in self._boxes:
            xc = (r.x() + r.width()/2.0) / w
            yc = (r.y() + r.height()/2.0) / h
            nw = r.width() / w
            nh = r.height() / h
            # clamp
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            yolo_boxes.append((current_class_id, xc, yc, nw, nh))
            
        return yolo_boxes

    def paintEvent(self, event):
        # Draw scaled pixmap
        painter = QPainter(self)
        
        if self._pixmap.isNull():
            return
            
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Calculate offset
        x_off = (self.width() - scaled_pixmap.width()) // 2
        y_off = (self.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x_off, y_off, scaled_pixmap)
        
        # Draw boxes
        pen = QPen(QColor("#00d4ff"), 2)
        painter.setPen(pen)
        
        scale_x = scaled_pixmap.width() / self._pixmap.width()
        scale_y = scaled_pixmap.height() / self._pixmap.height()
        
        for r in self._boxes:
            scaled_r = QRectF(x_off + r.x() * scale_x, y_off + r.y() * scale_y, r.width() * scale_x, r.height() * scale_y)
            painter.drawRect(scaled_r)
            
        if self._current_rect:
            scaled_r = QRectF(x_off + self._current_rect.x() * scale_x, y_off + self._current_rect.y() * scale_y, self._current_rect.width() * scale_x, self._current_rect.height() * scale_y)
            painter.drawRect(scaled_r)
            
    def _map_to_image(self, pos):
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_off = (self.width() - scaled_pixmap.width()) // 2
        y_off = (self.height() - scaled_pixmap.height()) // 2
        
        scale_x = self._pixmap.width() / scaled_pixmap.width()
        scale_y = self._pixmap.height() / scaled_pixmap.height()
        
        ix = (pos.x() - x_off) * scale_x
        iy = (pos.y() - y_off) * scale_y
        
        # clamp
        ix = max(0, min(self._pixmap.width(), ix))
        iy = max(0, min(self._pixmap.height(), iy))
        
        return QPointF(ix, iy)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_pos = self._map_to_image(event.position())
            self._current_rect = QRectF(self._start_pos, self._start_pos)
            self.update()

    def mouseMoveEvent(self, event):
        if self._start_pos:
            end_pos = self._map_to_image(event.position())
            self._current_rect = QRectF(self._start_pos, end_pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._current_rect:
            if self._current_rect.width() > 5 and self._current_rect.height() > 5:
                self._boxes.append(self._current_rect)
            self._current_rect = None
            self._start_pos = None
            self.update()
