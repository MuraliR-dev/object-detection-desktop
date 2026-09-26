import cv2
import numpy as np
import random
import os

def augment_image_with_bboxes(img, bboxes):
    """
    Apply a random augmentation to an image and adjust its bounding boxes.
    bboxes: list of tuples (class_id, x_center, y_center, width, height) in YOLO format (0.0 to 1.0)
    Returns: (augmented_img, augmented_bboxes)
    """
    h, w = img.shape[:2]
    
    aug_type = random.randint(0, 3)
    
    aug_img = img.copy()
    aug_bboxes = []
    
    if aug_type == 0: # Flip Horizontal
        aug_img = cv2.flip(img, 1)
        for bbox in bboxes:
            cls_id, xc, yc, bw, bh = bbox
            aug_bboxes.append((cls_id, 1.0 - xc, yc, bw, bh))
            
    elif aug_type == 1: # Brightness/Contrast
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        aug_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        aug_bboxes = bboxes.copy()
        
    elif aug_type == 2: # Slight scaling/cropping (zoom in slightly)
        scale = random.uniform(1.05, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        x_start = random.randint(0, new_w - w)
        y_start = random.randint(0, new_h - h)
        
        aug_img = resized[y_start:y_start+h, x_start:x_start+w]
        
        for bbox in bboxes:
            cls_id, xc, yc, bw, bh = bbox
            px_xc = xc * w
            px_yc = yc * h
            px_w = bw * w
            px_h = bh * h
            
            new_px_xc = px_xc * scale
            new_px_yc = px_yc * scale
            new_px_w = px_w * scale
            new_px_h = px_h * scale
            
            final_px_xc = new_px_xc - x_start
            final_px_yc = new_px_yc - y_start
            
            if (final_px_xc + new_px_w/2 < 0 or final_px_xc - new_px_w/2 > w or
                final_px_yc + new_px_h/2 < 0 or final_px_yc - new_px_h/2 > h):
                continue
                
            x_min = max(0, final_px_xc - new_px_w/2)
            x_max = min(w, final_px_xc + new_px_w/2)
            y_min = max(0, final_px_yc - new_px_h/2)
            y_max = min(h, final_px_yc + new_px_h/2)
            
            if x_max <= x_min or y_max <= y_min:
                continue
                
            final_xc = ((x_min + x_max) / 2.0) / w
            final_yc = ((y_min + y_max) / 2.0) / h
            final_bw = (x_max - x_min) / w
            final_bh = (y_max - y_min) / h
            
            aug_bboxes.append((cls_id, final_xc, final_yc, final_bw, final_bh))
            
    elif aug_type == 3: # Noise or Blur
        if random.random() > 0.5:
            k = random.choice([3, 5])
            aug_img = cv2.GaussianBlur(img, (k, k), 0)
        else:
            noise = np.zeros(img.shape, np.int16)
            cv2.randn(noise, 0, 15)
            aug_img = cv2.add(img, noise, dtype=cv2.CV_8UC3)
        aug_bboxes = bboxes.copy()
        
    return aug_img, aug_bboxes

def read_yolo_labels(txt_path):
    bboxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return bboxes

def write_yolo_labels(txt_path, bboxes):
    with open(txt_path, 'w') as f:
        for bbox in bboxes:
            f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
