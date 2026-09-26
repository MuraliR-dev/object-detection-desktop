# Object Detection Studio

## PROJECT NAME
Object Detection Studio

## PROJECT PURPOSE
This project is a desktop YOLO Custom Object Detection application. The application allows users to run real-time inference using standard YOLO models and easily train custom models for new objects using an intuitive graphical user interface. The primary final detection mode is real-time webcam detection.

## FEATURES
- **Real-time Webcam Detection**: Dual-model support (Default COCO + Custom trained models).
- **Custom Object Training**: Create a dataset, annotate images, and train a YOLO model within the application.
- **Dataset Management**: Add classes and images, annotate them with bounding boxes, and validate/prepare the dataset automatically.
- **Data Augmentation**: Automatically augment small datasets to improve model accuracy.
- **Hot-swapping Models**: Seamlessly load the newly trained `best.pt` custom model.
- **Clean UI**: Modern, dark-themed interface built with PySide6.

## TECHNOLOGIES
- **Python 3.8+**
- **PySide6** (Qt for Python GUI)
- **Ultralytics YOLO11** (Inference and Training engine)
- **OpenCV** (Webcam capture and drawing utilities)
- **PyTorch** (Backend for YOLO)
- **NumPy & Pillow** (Image processing)

## REQUIREMENTS
- A modern Windows environment.
- Python 3.8 or newer.
- A functional webcam.

## INSTALLATION

### VIRTUAL ENVIRONMENT SETUP & DEPENDENCY INSTALLATION

Open a PowerShell or Command Prompt in the project directory:

```bat
cd D:\objectDetaction
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify dependencies by running:
```bat
python -c "import PySide6, cv2, ultralytics, torch; print('Dependencies OK')"
```

## PROJECT STRUCTURE
```
D:\objectDetaction
├── main.py                     # Main application entry point
├── run.py                      # Canonical entry wrapper
├── requirements.txt            # Python dependencies
├── models/
│   └── yolo11n.pt              # Base pretrained model (DO NOT DELETE)
├── detector/
│   ├── core.py                 # Core detection logic (ObjectDetector)
│   ├── config.py               # Constants and configuration
│   └── utils.py                # Drawing and utility functions
├── gui/
│   ├── main_window.py          # Main PySide6 Window
│   ├── training_dialog.py      # Custom Object Training Dialog & Worker
│   ├── annotation_dialog.py    # Bounding Box Annotation UI
│   ├── augmentation.py         # Dataset Augmentation logic
│   ├── workers.py              # Webcam background thread
│   ├── widgets.py              # Custom UI elements
│   └── theme.py                # Application styling
└── datasets/                   # Generated custom training datasets
```

## HOW TO RUN

To start the application, use the virtual environment:

```bat
cd D:\objectDetaction
.\venv\Scripts\activate
python run.py
```

## CUSTOM MODEL TRAINING WORKFLOW
1. **Start application**: Run `python run.py`.
2. **Open Custom Object Training**: Click the "Add / Train New Object" button.
3. **Enter project name**: Set a name for your project in the "PROJECT SETTINGS" section.
4. **Add class**: In the "CLASSES" section, type a class name (e.g., `atm_card`) and click "Add Class".
5. **Add images**: In the "DATASET" section, click "Add Images" to select `.jpg` or `.png` files.

## ANNOTATION WORKFLOW
6. **Select image**: Click on an unannotated image in the dataset list.
7. **Annotate**: Click "Annotate Selected Image". Select the class from the dropdown and draw a bounding box around the object.
8. **Save annotations**: Click "Save Annotation" to generate the YOLO format label file.

## DATASET VALIDATION
9. **Validate Dataset**: Click "Validate Dataset". The system checks for empty classes, images without annotations, and invalid bounding boxes.

## DATASET PREPARATION
10. **Prepare Dataset**: Click "Prepare Dataset". This splits your data into `train` and `val` folders and generates a `data.yaml` file. If the dataset is small, augmentation will be applied.

## TRAINING
11. **Set Training Settings**: Adjust Epochs, Image Size, and Batch Size.
12. **Start Training**: Click "Start Training". The application will train the custom model using `yolo11n.pt` as a base.
13. **Wait for training**: Watch the log. It runs on a background thread so the UI remains responsive.

## BEST.PT LOADING
14. **Verify best.pt**: Once training completes, the exact `best.pt` file from the current run is identified.
15. **Load best.pt**: Click "Load best.pt" (or accept the dialog prompt) to load the new custom model into the application.

## WEBCAM DETECTION
16. **Start Webcam**: From the main window, select your camera and click "Start Webcam".
17. **Test custom object**: Hold your custom object in front of the webcam. Both default COCO objects (like `person`) and your custom object will be detected simultaneously.

## TROUBLESHOOTING

- **PySide6 missing or dependency errors**:
  Ensure you are using the virtual environment:
  ```bat
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```

- **Webcam not opening**:
  Check camera permissions in Windows settings. Ensure no other applications (like Zoom or Teams) are locking the camera. Try changing the "Camera" index to 1 or 2.

- **Missing yolo11n.pt**:
  The application requires the pretrained model to start. Place `yolo11n.pt` inside the `models/` directory.

- **Old custom model appearing**:
  Ensure old `runs/` or `datasets/` folders were deleted, and verify that the newly generated `best.pt` was loaded by checking the printed "Classes" in the console.

- **Training fails**:
  Check the console logs for detailed error messages. Ensure that your dataset was successfully validated and prepared. Memory errors may require reducing the "Batch Size" in training settings.

- **No custom detection**:
  Verify the current model classes by checking `model.names` in the console. Ensure your model learned the features by increasing the dataset size and training for more epochs.
