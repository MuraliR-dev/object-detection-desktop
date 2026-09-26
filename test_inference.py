import os
import cv2
from ultralytics import YOLO

def test_inference():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find latest best.pt
    from detector.core import find_latest_best_pt
    best_pt = find_latest_best_pt(project_root)
    
    if not best_pt:
        print("No best.pt found!")
        return
        
    print(f"Loading custom model: {best_pt}")
    model = YOLO(best_pt)
    print(f"Classes: {model.names}")
    
    # Find a validation image
    val_dir = os.path.join(project_root, "datasets", "my_object", "images", "val")
    if not os.path.isdir(val_dir):
        print(f"Val dir not found: {val_dir}")
        return
        
    images = os.listdir(val_dir)
    if not images:
        print("No images in val dir!")
        return
        
    img_path = os.path.join(val_dir, images[0])
    print(f"Testing on image: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image")
        return
        
    res = model(img, conf=0.1)
    
    for box in res[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        name = model.names.get(cls_id, "unknown")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        print(f"Detected: {name} (conf: {conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
        
if __name__ == "__main__":
    test_inference()
