"""
gui/training_dialog.py
----------------------
Custom Object Training Dialog.
Manages dataset creation, validation, annotations, and YOLO training in a QThread.
"""
import os
import shutil
import random

# Project root is one level up from gui/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QGroupBox, QSpinBox, QProgressBar,
    QFormLayout, QComboBox, QSplitter, QCheckBox, QScrollArea, QWidget, QGridLayout, QSizePolicy
)
from gui.annotation_dialog import AnnotationDialog
import cv2
try:
    from gui.augmentation import augment_image_with_bboxes, read_yolo_labels, write_yolo_labels
except ImportError:
    pass # Will handle gracefully if missing

class TrainingWorker(QThread):
    progress = Signal(str)
    finished = Signal(str) # returns best.pt path if successful, empty if failed
    
    def __init__(self, data_yaml, epochs, imgsz, batch, device, project_name=None, class_names=None):
        super().__init__()
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.project_name = project_name or ""
        self.class_names = class_names or []
        
    def run(self):
        try:
            from ultralytics import YOLO

            # Resolve base model to absolute path
            base_path = os.path.join(_PROJECT_ROOT, "models", "yolo11n.pt")
            if not os.path.isfile(base_path):
                base_path = "models/yolo11n.pt"  # fallback to relative

            print(f"\n[TRAIN] ============================================")
            print(f"[TRAIN] Project:    {self.project_name}")
            print(f"[TRAIN] Classes:    {self.class_names}")
            print(f"[TRAIN] Data YAML:  {self.data_yaml}")
            print(f"[TRAIN] Base Model: {base_path}")
            print(f"[TRAIN] Epochs:     {self.epochs}")
            print(f"[TRAIN] Batch:      {self.batch}")
            print(f"[TRAIN] Device:     {self.device or 'auto'}")
            print(f"[TRAIN] ============================================\n")

            self.progress.emit("Validating master dataset before training...")
            import yaml
            with open(self.data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                num_classes = len(data.get('names', []))
                
            base_dir = os.path.dirname(self.data_yaml)
            invalid_labels = 0
            total_images = 0
            class_counts = {i: 0 for i in range(num_classes)}
            
            for split in ['train', 'val']:
                img_dir = os.path.join(base_dir, 'images', split)
                lbl_dir = os.path.join(base_dir, 'labels', split)
                if os.path.isdir(img_dir):
                    for img_f in os.listdir(img_dir):
                        if img_f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            total_images += 1
                            lbl_f = os.path.splitext(img_f)[0] + '.txt'
                            lbl_p = os.path.join(lbl_dir, lbl_f)
                            if not os.path.exists(lbl_p):
                                invalid_labels += 1
                            else:
                                with open(lbl_p, 'r') as f:
                                    for line in f.readlines():
                                        parts = line.strip().split()
                                        if len(parts) == 5:
                                            cid = int(float(parts[0]))
                                            if cid < 0 or cid >= num_classes:
                                                invalid_labels += 1
                                            elif cid in class_counts:
                                                class_counts[cid] += 1
                                                
            if invalid_labels > 0:
                raise Exception(f"Dataset validation failed: Found {invalid_labels} invalid labels out of bounds out of {total_images} images.")
            
            print("\n" + "="*50)
            print("MASTER DATASET VALIDATION & TRAINING LOG")
            print("="*50)
            print(f"Master Dataset Path: {base_dir}")
            print(f"Total Classes: {num_classes}")
            print(f"Total Images: {total_images}")
            print("Class Mapping & Distribution:")
            if isinstance(data.get('names'), dict):
                names_dict = data['names']
            else:
                names_dict = {i: n for i, n in enumerate(data.get('names', []))}
            for cid, cname in names_dict.items():
                print(f"  ID {cid}: '{cname}' -> {class_counts.get(cid, 0)} instances")
            print("="*50 + "\n")

            self.progress.emit(f"Loading base model: {base_path}")
            model = YOLO(base_path)
            
            self.progress.emit("Starting training... (Check console for detailed logs)")
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device,
                patience=0,      # Disable early stopping — fires prematurely on small datasets
                workers=0,       # CRITICAL for Windows CPU: avoids multiprocessing spawn errors
                cache=False,     # Avoid stale cached data from previous runs
                exist_ok=True,
            )
            
            # Locate best.pt — try results.save_dir first, then fallback
            best_pt = None
            try:
                candidate = str(results.save_dir / "weights" / "best.pt")
                if os.path.isfile(candidate):
                    best_pt = candidate
            except Exception:
                pass

            if not best_pt:
                # Fallback: scan runs/ tree for newest best.pt
                from detector.core import find_latest_best_pt
                best_pt = find_latest_best_pt(_PROJECT_ROOT)

            if best_pt and os.path.isfile(best_pt):
                print(f"\n[TRAIN] ============================================")
                print(f"[TRAIN] Best Model (from Ultralytics): {best_pt}")
                
                try:
                    verify = YOLO(best_pt)
                    print(f"[TRAIN] Classes:    {verify.names}")
                    print(f"[TRAIN] Num Classes:{len(verify.names)}")

                    # === POST-TRAINING CONFIDENCE TEST ===
                    # Test best.pt on a val image to see if it actually works.
                    # If max_conf < 0.1, Ultralytics selected a weak early epoch
                    # as 'best' by fitness score. Fall back to last.pt instead.
                    import cv2 as _cv2
                    val_dir = os.path.join(_PROJECT_ROOT, "datasets", self.project_name, "images", "val")
                    best_max_conf = 0.0
                    if os.path.isdir(val_dir):
                        val_imgs_list = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        for _vi in val_imgs_list:
                            _img = _cv2.imread(os.path.join(val_dir, _vi))
                            if _img is not None:
                                _res = verify(_img, conf=0.01, verbose=False)
                                for _b in _res[0].boxes:
                                    best_max_conf = max(best_max_conf, float(_b.conf[0]))
                    
                    print(f"[TRAIN] best.pt max_conf on val images: {best_max_conf:.4f}")
                    
                    if best_max_conf < 0.10:
                        # best.pt is too weak — Ultralytics picked a bad early epoch.
                        # Try last.pt (final epoch weights) instead.
                        save_dir = None
                        try:
                            save_dir = str(results.save_dir)
                        except Exception:
                            pass
                        last_candidates = []
                        if save_dir:
                            last_candidates.append(os.path.join(save_dir, "weights", "last.pt"))
                        # Also check all runs
                        import re
                        for root, dirs, files in os.walk(os.path.join(_PROJECT_ROOT, "runs")):
                            if "last.pt" in files:
                                last_candidates.append(os.path.join(root, "last.pt"))
                        last_candidates.sort(key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0, reverse=True)
                        
                        switched = False
                        for last_pt in last_candidates:
                            if not os.path.isfile(last_pt):
                                continue
                            last_model = YOLO(last_pt)
                            last_max_conf = 0.0
                            if os.path.isdir(val_dir):
                                for _vi in val_imgs_list:
                                    _img = _cv2.imread(os.path.join(val_dir, _vi))
                                    if _img is not None:
                                        _res = last_model(_img, conf=0.01, verbose=False)
                                        for _b in _res[0].boxes:
                                            last_max_conf = max(last_max_conf, float(_b.conf[0]))
                            print(f"[TRAIN] last.pt max_conf on val images: {last_max_conf:.4f}")
                            if last_max_conf > best_max_conf:
                                print(f"[TRAIN] SWITCHING to last.pt (confidence {last_max_conf:.4f} > {best_max_conf:.4f})")
                                best_pt = last_pt
                                verify = last_model
                                best_max_conf = last_max_conf
                                switched = True
                                break
                        
                        if switched:
                            print(f"[TRAIN] Using last.pt: {best_pt}")
                        else:
                            print(f"[TRAIN] WARNING: Both best.pt and last.pt have low confidence — dataset may be insufficient.")
                    
                    val_count = 0
                    try:
                        val_check = os.path.join(_PROJECT_ROOT, "datasets", self.project_name, "images", "val")
                        if os.path.isdir(val_check):
                            val_count = len([f for f in os.listdir(val_check) if f.lower().endswith(('.jpg','.jpeg','.png'))])
                    except Exception:
                        pass
                    if val_count < 5:
                        print(f"[TRAIN] WARNING: Only {val_count} val images — metrics may not reflect real-world accuracy.")

                except Exception as ve:
                    print(f"[TRAIN] Warning: could not verify — {ve}")
                    import traceback; traceback.print_exc()
                print(f"[TRAIN] ============================================")
                print(f"[TRAIN] Final model path: {best_pt}")
                print(f"[TRAIN] ============================================\n")
                self.progress.emit("Training completed successfully!")
                self.finished.emit(best_pt)
            else:
                self.progress.emit("Error: best.pt not found after training.")
                self.finished.emit("")
                
        except Exception as e:
            import traceback
            print(f"[TRAIN] EXCEPTION: {e}")
            traceback.print_exc()
            self.progress.emit(f"Training failed: {e}")
            self.finished.emit("")


class TrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Object Training")
        self.setMinimumSize(1200, 750)
        
        # Center the window
        if parent:
            self.move(parent.geometry().center() - self.rect().center())
            
        custom_style = """
        QDialog {
            background-color: #0d1117;
        }
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QGroupBox {
            background: #161b22;
            border: 1px solid #303846;
            border-radius: 10px;
            padding: 16px 12px 12px 12px;
            margin-top: 0px;
        }
        QLineEdit, QSpinBox, QComboBox {
            background-color: #171c25;
            border: 1px solid #303846;
            border-radius: 6px;
            padding: 8px;
            color: #ffffff;
            font-size: 13px;
            selection-background-color: #00d9ff;
            selection-color: #000000;
        }
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
            border: 1px solid #00d9ff;
            background-color: #0d1117;
        }
        QLineEdit::placeholder {
            color: #7f8a9a;
        }
        QListWidget {
            background-color: #171c25;
            border: 1px solid #303846;
            border-radius: 6px;
            color: #ffffff;
            font-size: 13px;
            padding: 4px;
            outline: none;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #21262d;
        }
        QListWidget::item:selected {
            background-color: #00d9ff;
            color: #000000;
            border-radius: 4px;
        }
        QPushButton {
            background-color: #21262d;
            border: 1px solid #363b42;
            border-radius: 6px;
            color: #ffffff;
            padding: 10px 16px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #30363d;
            border: 1px solid #00d9ff;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #00d9ff;
            color: #000000;
            border: none;
        }
        QLabel {
            color: #b8c0cc;
            font-size: 13px;
        }
        QCheckBox {
            color: #b8c0cc;
            font-size: 13px;
        }
        QMessageBox {
            background-color: #0d1117;
        }
        QMessageBox QLabel {
            color: #ffffff;
            font-size: 13px;
        }
        """
        self.setStyleSheet(custom_style)
        
        self.best_model_path = None
        self._build_ui()
        
    def _create_section_header(self, title):
        lbl = QLabel(title)
        lbl.setStyleSheet("background-color: transparent; color: #00d9ff; font-size: 16px; font-weight: 600; padding: 0px; margin: 0px;")
        lbl.setMinimumHeight(28)
        lbl.setMaximumHeight(36)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return lbl

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        
        content_widget = QWidget()
        scroll.setWidget(content_widget)
        
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 01 PROJECT SETTINGS
        layout.addWidget(self._create_section_header("01  PROJECT SETTINGS"))
        layout.addSpacing(8)
        proj_group = QGroupBox()
        proj_layout = QHBoxLayout(proj_group)
        proj_label = QLabel("Project Name:")
        proj_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.proj_name = QLineEdit("my_object")
        proj_layout.addWidget(proj_label)
        proj_layout.addWidget(self.proj_name, 1)
        layout.addWidget(proj_group)
        layout.addSpacing(20)
        
        # Grid for Classes and Dataset
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(8)
        grid_layout.setHorizontalSpacing(20)
        
        # 02 CLASSES
        grid_layout.addWidget(self._create_section_header("02  CLASSES"), 0, 0)
        class_group = QGroupBox()
        class_layout = QVBoxLayout(class_group)
        
        class_input_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Class name (e.g. bottle)")
        btn_add_class = QPushButton("Add Class")
        btn_add_class.clicked.connect(self._add_class)
        class_input_layout.addWidget(self.class_input)
        class_input_layout.addWidget(btn_add_class)
        
        self.class_list = QListWidget()
        btn_rem_class = QPushButton("Remove Selected Class")
        btn_rem_class.clicked.connect(lambda: self.class_list.takeItem(self.class_list.currentRow()))
        
        class_layout.addLayout(class_input_layout)
        class_layout.addWidget(self.class_list)
        class_layout.addWidget(btn_rem_class)
        grid_layout.addWidget(class_group, 1, 0)
        
        # 03 DATASET (IMAGES)
        grid_layout.addWidget(self._create_section_header("03  DATASET (IMAGES)"), 0, 1)
        data_group = QGroupBox()
        data_layout = QVBoxLayout(data_group)
        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(200)
        
        btn_add_img = QPushButton("Add Images")
        btn_add_img.clicked.connect(self._add_images)
        btn_rem_img = QPushButton("Remove Selected")
        btn_rem_img.clicked.connect(lambda: self.image_list.takeItem(self.image_list.currentRow()))
        btn_annotate = QPushButton("Annotate Selected Image")
        btn_annotate.clicked.connect(self._annotate_image)
        
        data_layout.addWidget(self.image_list)
        data_btn_layout = QHBoxLayout()
        data_btn_layout.addWidget(btn_add_img)
        data_btn_layout.addWidget(btn_rem_img)
        data_btn_layout.addWidget(btn_annotate)
        data_layout.addLayout(data_btn_layout)
        grid_layout.addWidget(data_group, 1, 1)
        
        layout.addLayout(grid_layout)
        layout.addSpacing(20)
        
        # 04 VALIDATE / PREPARE
        layout.addWidget(self._create_section_header("04  VALIDATE / PREPARE"))
        layout.addSpacing(8)
        prep_group = QGroupBox()
        prep_layout = QVBoxLayout(prep_group)
        self.lbl_prep_status = QLabel("Status: Not validated")
        self.lbl_prep_status.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px;")
        self.lbl_small_dataset = QLabel("")
        self.lbl_small_dataset.setStyleSheet("color: #ffb86c; font-weight: bold;")
        
        btn_validate = QPushButton("Validate Dataset")
        btn_validate.clicked.connect(self._validate_dataset)
        
        prep_form = QHBoxLayout()
        self.check_augment = QCheckBox("Enable Augmentation")
        self.check_augment.setChecked(True)
        lbl_aug = QLabel("Augment Multiplier:")
        self.spin_augment_multiplier = QSpinBox()
        self.spin_augment_multiplier.setRange(1, 20)
        self.spin_augment_multiplier.setValue(5)
        prep_form.addWidget(self.check_augment)
        prep_form.addSpacing(20)
        prep_form.addWidget(lbl_aug)
        prep_form.addWidget(self.spin_augment_multiplier)
        prep_form.addStretch()
        
        btn_prepare = QPushButton("Prepare Dataset")
        btn_prepare.clicked.connect(self._prepare_dataset)
        
        prep_layout.addWidget(self.lbl_prep_status)
        prep_layout.addWidget(self.lbl_small_dataset)
        prep_layout.addSpacing(10)
        prep_layout.addWidget(btn_validate)
        prep_layout.addLayout(prep_form)
        prep_layout.addWidget(btn_prepare)
        layout.addWidget(prep_group)
        layout.addSpacing(20)
        
        # 05 TRAINING SETTINGS
        layout.addWidget(self._create_section_header("05  TRAINING SETTINGS"))
        layout.addSpacing(8)
        train_group = QGroupBox()
        train_layout = QHBoxLayout(train_group)
        
        lbl_epochs = QLabel("Epochs:")
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 1000); self.spin_epochs.setValue(50)
        
        lbl_imgsz = QLabel("Image Size:")
        self.spin_imgsz = QSpinBox(); self.spin_imgsz.setRange(32, 2048); self.spin_imgsz.setValue(640); self.spin_imgsz.setSingleStep(32)
        
        lbl_batch = QLabel("Batch Size:")
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(1, 256); self.spin_batch.setValue(8)
        
        lbl_device = QLabel("Device:")
        self.combo_device = QComboBox(); self.combo_device.addItems(["", "cpu", "0"])
        
        train_layout.addWidget(lbl_epochs)
        train_layout.addWidget(self.spin_epochs)
        train_layout.addSpacing(20)
        train_layout.addWidget(lbl_imgsz)
        train_layout.addWidget(self.spin_imgsz)
        train_layout.addSpacing(20)
        train_layout.addWidget(lbl_batch)
        train_layout.addWidget(self.spin_batch)
        train_layout.addSpacing(20)
        train_layout.addWidget(lbl_device)
        train_layout.addWidget(self.combo_device)
        train_layout.addStretch()
        layout.addWidget(train_group)
        layout.addSpacing(20)
        
        # 06 EXECUTION
        layout.addWidget(self._create_section_header("06  EXECUTION"))
        layout.addSpacing(8)
        exec_group = QGroupBox()
        exec_layout = QVBoxLayout(exec_group)
        
        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self._start_training)
        
        self.lbl_train_log = QLabel("Ready to train.")
        self.lbl_train_log.setWordWrap(True)
        self.lbl_train_log.setStyleSheet("color: #00d9ff; font-weight: bold;")
        
        self.btn_load_best = QPushButton("Load best.pt")
        self.btn_load_best.setEnabled(False)
        self.btn_load_best.clicked.connect(self.accept)
        
        exec_layout.addWidget(self.btn_train)
        exec_layout.addWidget(self.lbl_train_log)
        exec_layout.addWidget(self.btn_load_best)
        layout.addWidget(exec_group)
        layout.addSpacing(20)
        
        layout.addStretch()
        
    def _add_class(self):
        cname = self.class_input.text().strip()
        if not cname: return
        # Check duplicate
        for i in range(self.class_list.count()):
            if self.class_list.item(i).text() == cname:
                return
        self.class_list.addItem(cname)
        self.class_input.clear()
        
    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.jpg *.jpeg *.png)")
        if paths:
            for p in paths:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.UserRole, p)
                txt_path = os.path.splitext(p)[0] + ".txt"
                if os.path.exists(txt_path):
                    item.setText(f"✓ {os.path.basename(p)}")
                else:
                    item.setText(f"⚠ {os.path.basename(p)} (Not Annotated)")
                self.image_list.addItem(item)
                
    def _annotate_image(self):
        item = self.image_list.currentItem()
        if not item: return
        if self.class_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Add at least one class first!")
            return
            
        classes = [self.class_list.item(i).text() for i in range(self.class_list.count())]
        img_path = item.data(Qt.UserRole)
        dialog = AnnotationDialog(img_path, classes, self)
        dialog.exec()
        
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            item.setText(f"✓ {os.path.basename(img_path)}")
        else:
            item.setText(f"⚠ {os.path.basename(img_path)} (Not Annotated)")
        
    def _validate_dataset(self):
        if self.class_list.count() == 0:
            self.lbl_prep_status.setText("Status: Validation Failed (No classes defined)")
            QMessageBox.warning(self, "Validation Failed", "Please add at least one class.")
            return False
            
        if self.image_list.count() == 0:
            self.lbl_prep_status.setText("Status: Validation Failed (No images added)")
            QMessageBox.warning(self, "Validation Failed", "Please add images.")
            return False
            
        annotated_count = 0
        valid_pairs = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            img_path = item.data(Qt.UserRole)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                # rigorous check if valid
                is_valid = False
                try:
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    # check bounds
                                    cls_id, xc, yc, w, h = map(float, parts)
                                    if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1:
                                        is_valid = True
                except: pass
                
                if is_valid:
                    annotated_count += 1
                    valid_pairs.append((img_path, txt_path))
                    item.setText(f"✓ {os.path.basename(img_path)}")
                else:
                    item.setText(f"❌ {os.path.basename(img_path)} (Invalid box)")
            else:
                item.setText(f"⚠ {os.path.basename(img_path)} (Not Annotated)")
                
        if annotated_count < 2:
            self.lbl_prep_status.setText("Status: Validation Failed (Need >= 2 annotated images)")
            QMessageBox.warning(self, "Validation Failed", f"Dataset validation failed.\nAt least 2 annotated images are required (1 for training, 1 for validation).\nFound: {annotated_count}")
            return False
            
        self.lbl_prep_status.setText(f"Status: Dataset Valid\nImages: {self.image_list.count()}  |  Valid annotations: {annotated_count}")
        
        # Small dataset logic
        if annotated_count <= 10:
            self.lbl_small_dataset.setText(f"⚠ Small dataset: {annotated_count} images.\nAugmentation is recommended.")
            self.check_augment.setChecked(True)
            self.spin_epochs.setValue(100) # Give it more time to learn
            self.spin_batch.setValue(4)    # Small batch for small dataset
        else:
            self.lbl_small_dataset.setText("")
            
        self._valid_pairs = valid_pairs
        return True

    def _prepare_dataset(self):
        if not self._validate_dataset():
            return
            
        if not hasattr(self, '_valid_pairs') or not self._valid_pairs:
            return
            
        # 1. ALWAYS USE MASTER DATASET
        base_dir = os.path.join(_PROJECT_ROOT, "datasets", "daily_objects")
        yaml_path = os.path.join(base_dir, "data.yaml")
        
        master_classes = {} # mapping master_id -> class_name
        
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        if isinstance(data['names'], dict):
                            master_classes = {int(k): v for k, v in data['names'].items()}
                        elif isinstance(data['names'], list):
                            master_classes = {i: v for i, v in enumerate(data['names'])}
            except Exception as e:
                print("Could not parse existing data.yaml, starting fresh.", e)
        
        # Build list of classes
        master_class_list = [master_classes.get(i, f"class_{i}") for i in range(len(master_classes) if master_classes else 0)]
        
        local_classes = [self.class_list.item(i).text() for i in range(self.class_list.count())]
        
        class_mapping = {}
        for local_idx, cname in enumerate(local_classes):
            if cname not in master_class_list:
                master_class_list.append(cname)
            class_mapping[local_idx] = master_class_list.index(cname)
            
        # Create directories without removing old ones!
        for split in ["train", "val"]:
            for kind in ["images", "labels"]:
                d = os.path.join(base_dir, kind, split)
                os.makedirs(d, exist_ok=True)

        pairs = self._valid_pairs
        
        # Safe Split: guaranteed at least 1 validation image
        random.shuffle(pairs)
        if len(pairs) <= 5:
            split_idx = len(pairs) - 1
        else:
            split_idx = int(len(pairs) * 0.8)
            
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        
        do_augment = self.check_augment.isChecked()
        aug_mult = self.spin_augment_multiplier.value()
        
        total_train = 0
        total_val = 0

        # Helper to read, map, and write labels
        def map_and_copy_label(src_txt, dst_txt):
            try:
                with open(src_txt, 'r') as f:
                    lines = f.readlines()
                with open(dst_txt, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            old_id = int(float(parts[0]))
                            new_id = class_mapping.get(old_id, old_id) # map to new ID
                            f.write(f"{new_id} {' '.join(parts[1:])}\n")
            except Exception as e:
                print(f"Failed to map label {src_txt}: {e}")

        # Process Validation (NO AUGMENTATION EVER)
        for img_path, txt_path in val_pairs:
            shutil.copy2(img_path, os.path.join(base_dir, "images", "val", os.path.basename(img_path)))
            map_and_copy_label(txt_path, os.path.join(base_dir, "labels", "val", os.path.basename(txt_path)))
            total_val += 1
            
        # Process Training (WITH optional AUGMENTATION)
        for img_path, txt_path in train_pairs:
            shutil.copy2(img_path, os.path.join(base_dir, "images", "train", os.path.basename(img_path)))
            map_and_copy_label(txt_path, os.path.join(base_dir, "labels", "train", os.path.basename(txt_path)))
            total_train += 1
            
            if do_augment:
                try:
                    img = cv2.imread(img_path)
                    bboxes = read_yolo_labels(txt_path)
                    if img is not None and bboxes:
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        for i in range(aug_mult):
                            aug_img, aug_bboxes = augment_image_with_bboxes(img, bboxes)
                            if aug_bboxes:
                                out_img = os.path.join(base_dir, "images", "train", f"{base_name}_aug_{i}.jpg")
                                out_txt = os.path.join(base_dir, "labels", "train", f"{base_name}_aug_{i}.txt")
                                cv2.imwrite(out_img, aug_img)
                                
                                # Remap aug_bboxes before writing!
                                mapped_aug_bboxes = []
                                for b in aug_bboxes:
                                    old_id = int(b[0])
                                    new_id = class_mapping.get(old_id, old_id)
                                    mapped_aug_bboxes.append((new_id, b[1], b[2], b[3], b[4]))
                                    
                                write_yolo_labels(out_txt, mapped_aug_bboxes)
                                total_train += 1
                except Exception as e:
                    print(f"Augmentation failed for {img_path}: {e}")
                
        # data.yaml
        with open(yaml_path, 'w') as f:
            abs_base = os.path.abspath(base_dir).replace('\\', '/')
            f.write(f"path: {abs_base}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n\n")
            f.write("names:\n")
            for i, c in enumerate(master_class_list):
                f.write(f"  {i}: {c}\n")
                
        self.data_yaml_path = yaml_path
        self.lbl_prep_status.setText(f"Master Dataset ready.\nOriginals Added: {len(pairs)}\nTrain Samples Added: {total_train}\nVal Samples Added: {total_val}")
        QMessageBox.information(self, "Success", f"Master Dataset updated successfully!\nAdded Train: {total_train} samples\nAdded Val: {total_val} samples\ndata.yaml updated with {len(master_class_list)} classes.")
        
    def _start_training(self):
        if not hasattr(self, 'data_yaml_path') or not os.path.exists(self.data_yaml_path):
            QMessageBox.warning(self, "Warning", "Please prepare the dataset before starting training.")
            return
            
        self.btn_train.setEnabled(False)
        self.lbl_train_log.setText("Training started... (see console for live progress)")
        
        proj = self.proj_name.text().strip() or "custom_project"
        class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
        
        self.worker = TrainingWorker(
            self.data_yaml_path,
            self.spin_epochs.value(),
            self.spin_imgsz.value(),
            self.spin_batch.value(),
            self.combo_device.currentText(),
            project_name=proj,
            class_names=class_names,
        )
        self.worker.progress.connect(lambda msg: self.lbl_train_log.setText(msg))
        self.worker.finished.connect(self._training_finished)
        self.worker.start()
        
    def _training_finished(self, best_pt_path):
        self.btn_train.setEnabled(True)
        if best_pt_path:
            self.best_model_path = best_pt_path
            self.btn_load_best.setEnabled(True)
            self.lbl_train_log.setText(f"Training completed successfully.\nBest model: {best_pt_path}")
            QMessageBox.information(self, "Training Complete", f"Training completed successfully.\nBest model saved at:\n{best_pt_path}")
        else:
            QMessageBox.warning(self, "Training Failed", "Training failed. Check the console for details.")
