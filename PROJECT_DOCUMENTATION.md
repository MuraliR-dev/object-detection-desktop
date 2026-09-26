# PROJECT DOCUMENTATION
## Object Detection Studio — Real-Time AI Object Detection

**Course:** Computer Science / Artificial Intelligence  
**Technology Stack:** Python · YOLO11n · OpenCV · PySide6  
**Version:** 2.0

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Existing System](#5-existing-system)
6. [Proposed System](#6-proposed-system)
7. [System Requirements](#7-system-requirements)
8. [Software Requirements](#8-software-requirements)
9. [Hardware Requirements](#9-hardware-requirements)
10. [Technologies Used](#10-technologies-used)
11. [System Architecture](#11-system-architecture)
12. [System Workflow](#12-system-workflow)
13. [YOLO Architecture / Working Concept](#13-yolo-architecture--working-concept)
14. [Object Detection Pipeline](#14-object-detection-pipeline)
15. [OpenCV Processing](#15-opencv-processing)
16. [PySide6 GUI Architecture](#16-pyside6-gui-architecture)
17. [Image Detection](#17-image-detection)
18. [Webcam Detection](#18-webcam-detection)
19. [Video Detection](#19-video-detection)
20. [Object Counting](#20-object-counting)
21. [Confidence Threshold](#21-confidence-threshold)
22. [FPS](#22-fps)
23. [Custom Model Workflow](#23-custom-model-workflow)
24. [File Structure](#24-file-structure)
25. [Implementation Details](#25-implementation-details)
26. [Testing Methodology](#26-testing-methodology)
27. [Test Cases](#27-test-cases)
28. [Actual Test Results](#28-actual-test-results)
29. [Limitations](#29-limitations)
30. [Applications](#30-applications)
31. [Future Enhancements](#31-future-enhancements)
32. [Conclusion](#32-conclusion)

---

## 1. Abstract
Object Detection Studio is a real-time AI object detection application built with Python, PySide6, OpenCV, and Ultralytics YOLO11. It provides a highly responsive graphical user interface for detecting objects in live webcams, images, and videos. The system dynamically tracks FPS, counts objects per class, and allows dynamic model swapping, proving that state-of-the-art computer vision models can be packaged into accessible desktop software.

## 2. Introduction
Deep learning object detection typically requires command-line familiarity. Object Detection Studio brings YOLO11's cutting-edge speed and accuracy into a desktop GUI, allowing users to perform complex visual inference tasks intuitively. 

## 3. Problem Statement
Existing AI object detection implementations are often constrained to headless scripts or cloud APIs, making them inaccessible for non-programmers or offline use cases. A unified, fully offline desktop application that handles asynchronous inference without GUI lock-ups is necessary.

## 4. Objectives
- Integrate Ultralytics YOLO11n into a local desktop application.
- Support Image, Video, and live Webcam processing.
- Maintain a highly responsive GUI utilizing threaded workers.
- Provide live parameter control (confidence thresholds).
- Support loading custom trained YOLO models at runtime.

## 5. Existing System
Many systems rely on `cv2.imshow()` for rudimentary displays, which lack interactive controls like sliders, logs, and custom file dialogs, and block the main execution thread.

## 6. Proposed System
A PySide6-based application using `QThread` to isolate OpenCV capture and YOLO inference from the GUI event loop, ensuring the user interface remains smooth while performing heavy computational tasks in the background.

## 7. System Requirements
- OS: Windows 10/11, macOS, or Linux.
- Python: 3.10+ (Tested on 3.12.7).

## 8. Software Requirements
- ultralytics (v8.4.161)
- opencv-python (v5.0.0)
- PySide6 (v6.11.2)
- numpy (v2.5.3)

## 9. Hardware Requirements
- **CPU**: Quad-core processor recommended.
- **GPU**: Optional but recommended (NVIDIA CUDA support).
- **RAM**: Minimum 4GB.

## 10. Technologies Used
- Python for core logic.
- Ultralytics PyTorch implementation for YOLO.
- OpenCV for frame extraction and BGR matrix manipulation.
- PySide6 for Qt-based GUI rendering.

## 11. System Architecture
The system follows a Model-View separation:
- **Core Engine (`detector/`)**: Contains `ObjectDetector` which manages model weights and tensor inference.
- **Workers (`gui/workers.py`)**: `QThread` subclasses that capture frames, call the detector, and emit signals.
- **GUI (`gui/main_window.py`)**: Receives frame signals, updates `QPixmap` canvases, and manages user state.

## 12. System Workflow
User Input -> Main Thread -> Spawns QThread -> OpenCV Captures Frame -> YOLO Inference -> Bounding Boxes Drawn -> Frame Emitted to GUI -> UI Updates Display and Statistics.

## 13. YOLO Architecture / Working Concept
YOLO (You Only Look Once) frames object detection as a single regression problem. It divides the image into a grid and simultaneously predicts bounding boxes and class probabilities for each grid cell, passing through a convolutional neural network only once, enabling high-speed real-time detection.

## 14. Object Detection Pipeline
Frame captured -> Resized to 640x640 -> Normalized -> Passed to YOLO -> Non-Maximum Suppression (NMS) filters overlapping boxes -> Output mapped back to original dimensions -> Annotations drawn.

## 15. OpenCV Processing
OpenCV reads image files and captures video streams, representing frames as `numpy.ndarray` objects in BGR format. OpenCV drawing functions add rectangles and text overlays before passing the byte buffer to PySide6.

## 16. PySide6 GUI Architecture
A `QMainWindow` acts as the root. It contains a stretching `PreviewLabel` on the left and a fixed 320px sidebar on the right. The sidebar is wrapped in a `QScrollArea` to ensure responsiveness on small screens.

## 17. Image Detection
Uses `ImageWorker`. The application reads a selected image, runs a single inference pass, draws annotations, and saves the output to the `outputs/` folder.

## 18. Webcam Detection
Uses `WebcamWorker`. A `cv2.VideoCapture` loop continually reads frames. Thread safety is maintained via `QMutex` when accessing the confidence threshold.

## 19. Video Detection
Uses `VideoWorker`. Reads video frames sequentially, processes them, updates a `QProgressBar`, and writes the annotated frames to a new MP4 file using `cv2.VideoWriter`.

## 20. Object Counting
A dictionary dynamically aggregates class names and their occurrences for the current frame, updating the UI's "Objects" stat card and Detection Log list widget.

## 21. Confidence Threshold
A `QSlider` (5-95%) binds to a slot that updates the `conf_threshold` property in the worker threads, dynamically filtering out low-confidence predictions in real-time.

## 22. FPS
Calculated using a custom `FPSCounter` class that averages the time intervals of the last 30 frames, yielding an accurate and stable frame rate metric.

## 23. Custom Model Workflow
A dedicated "Load Custom .pt Model" button in the GUI allows users to swap the active YOLO model seamlessly without restarting the app.

Additionally, a comprehensive "Add / Train New Object" interface is included:
- **Dataset Creation**: Automatically scaffolds YOLO-compliant dataset structures (`images/train`, `labels/train`).
- **Annotation & YOLO Label Format**: Validates that users provide correct text files containing normalized `class_id center_x center_y width height`.
- **data.yaml**: Automatically generated dynamically based on defined custom classes.
- **Custom Training**: Launches a background `QThread` (`TrainWorker`) to execute Ultralytics YOLO training without freezing the PySide6 UI.
- **best.pt**: The application locates the optimal trained weights after epochs finish.
- **Custom Model Loading**: Instantly unloads the default model and loads `best.pt`, transitioning seamlessly into Image, Webcam, or Video detection modes with the newly taught classes.

## 24. File Structure
The project cleanly separates concerns into `detector/` (backend AI logic), `gui/` (frontend rendering), `models/` (weights), and `outputs/` (saved results).

## 25. Implementation Details
The codebase rigorously avoids blocking the Qt Main Event Loop. It correctly converts BGR matrices to RGB byte streams compatible with PySide6's `QImage` Format_RGB888.

## 26. Testing Methodology
Automated unit testing via `run_tests.py` verifies core detection logic without requiring a graphical display. Manual testing validates GUI responsivenes, layout integrity, and video output creation.

## 27. Test Cases
- Module Import & Instantiation
- Image Detection Output Generation
- Webcam Initialization
- Video File Batch Processing
- FPS Counter Accuracy
- Runtime Confidence Updates
- Graceful Error Handling (Missing Files)

## 28. Actual Test Results
`run_tests.py` reported 28/28 tests passed. The YOLO model loads in ~50-100ms. Synthetic test video processing ran at ~9 FPS on CPU. Missing files correctly raise `FileNotFoundError` handled gracefully by the GUI.

## 29. Limitations
- Does not support multi-GPU acceleration out of the box.
- Frame processing is CPU-bound on machines lacking CUDA, yielding lower FPS.
- Does not persist object IDs across frames (no tracking algorithms like DeepSORT).

## 30. Applications
- Security surveillance and perimeter monitoring.
- Educational demonstrations of deep learning principles.
- Prototype industrial visual inspection systems.

## 31. Future Enhancements
- Integration of object tracking algorithms.
- Export results to CSV/JSON format.

## 32. Conclusion
Object Detection Studio provides a robust, real-time, user-friendly interface for YOLO AI object detection. Its careful threading architecture and dynamic model loading capabilities make it highly adaptable while remaining accessible to non-technical end users.
