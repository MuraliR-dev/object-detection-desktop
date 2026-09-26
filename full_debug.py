"""
Full end-to-end diagnostic script.
Run with: venv\Scripts\python.exe full_debug.py
"""
import os, sys
sys.path.insert(0, '.')
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*60)
print("STEP 1: VERIFY best.pt")
print("="*60)

from ultralytics import YOLO

pt = os.path.join(ROOT, 'runs', 'detect', 'train-2', 'weights', 'best.pt')
print('[MODEL] Absolute path:', os.path.abspath(pt))
print('[MODEL] File exists:  ', os.path.isfile(pt))
print('[MODEL] File size:    ', os.path.getsize(pt) if os.path.isfile(pt) else 'N/A', 'bytes')

model = YOLO(pt)
print('[MODEL] Names:        ', model.names)
print('[MODEL] Task:         ', model.task)
print('[MODEL] Num classes:  ', len(model.names))

print("\n" + "="*60)
print("STEP 2: TRAINING CONFIGURATION (train-2/args.yaml)")
print("="*60)
args_yaml = os.path.join(ROOT, 'runs', 'detect', 'train-2', 'args.yaml')
try:
    import yaml
    with open(args_yaml) as f:
        args = yaml.safe_load(f)
    for k in ['model','data','epochs','imgsz','batch','workers','patience','device','optimizer']:
        print('  %-12s: %s' % (k, args.get(k, 'N/A')))
except Exception as e:
    print('  ERROR:', e)

print("\n" + "="*60)
print("STEP 3: TRAINING RESULTS CSV")
print("="*60)
results_csv = os.path.join(ROOT, 'runs', 'detect', 'train-2', 'results.csv')
try:
    with open(results_csv) as f:
        lines = f.readlines()
    # Print header + first 3 + last 5 rows
    header_cols = [c.strip() for c in lines[0].split(',')]
    
    # Find mAP50 and fitness cols
    def parse_row(line):
        return [x.strip() for x in line.split(',')]
    
    print('  First 3 epochs:')
    for line in lines[1:4]:
        cols = parse_row(line)
        row = dict(zip(header_cols, cols))
        epoch = row.get('                  epoch','?')
        box_loss = row.get('         train/box_loss','?')
        map50 = row.get('   metrics/mAP50(B)','?')
        map5095 = row.get('metrics/mAP50-95(B)','?')
        prec = row.get('   metrics/precision(B)','?')
        rec = row.get('      metrics/recall(B)','?')
        print(f'    epoch={epoch}  P={prec}  R={rec}  mAP50={map50}  mAP50-95={map5095}')
    print('  Last 5 epochs:')
    for line in lines[-5:]:
        cols = parse_row(line)
        row = dict(zip(header_cols, cols))
        epoch = row.get('                  epoch','?')
        map50 = row.get('   metrics/mAP50(B)','?')
        map5095 = row.get('metrics/mAP50-95(B)','?')
        prec = row.get('   metrics/precision(B)','?')
        rec = row.get('      metrics/recall(B)','?')
        print(f'    epoch={epoch}  P={prec}  R={rec}  mAP50={map50}  mAP50-95={map5095}')
except Exception as e:
    print('  ERROR:', e)
    import traceback; traceback.print_exc()

print("\n" + "="*60)
print("STEP 4: DATASET VERIFICATION")
print("="*60)
train_img_dir = os.path.join(ROOT, 'datasets', 'money_object', 'images', 'train')
train_lbl_dir = os.path.join(ROOT, 'datasets', 'money_object', 'labels', 'train')
val_img_dir   = os.path.join(ROOT, 'datasets', 'money_object', 'images', 'val')
val_lbl_dir   = os.path.join(ROOT, 'datasets', 'money_object', 'labels', 'val')

train_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
val_imgs   = [f for f in os.listdir(val_img_dir)   if f.lower().endswith(('.jpg','.jpeg','.png'))]
print('[DATASET] Train images:', len(train_imgs))
print('[DATASET] Val images:  ', len(val_imgs))

# Count labels
train_pos = 0
for lbl_f in os.listdir(train_lbl_dir):
    if lbl_f.endswith('.txt'):
        with open(os.path.join(train_lbl_dir, lbl_f)) as f:
            if f.read().strip():
                train_pos += 1
print('[DATASET] Train positive:', train_pos)
print('[DATASET] Train negative:', len(train_imgs) - train_pos)

print('[DATASET] Val labels:')
for lbl_f in sorted(os.listdir(val_lbl_dir)):
    if lbl_f.endswith('.txt'):
        with open(os.path.join(val_lbl_dir, lbl_f)) as f:
            content = f.read().strip()
        if content:
            for line in content.split('\n'):
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, xc, yc, w, h = [float(x) for x in parts]
                    print(f'  {lbl_f}: cls={int(cls_id)} xc={xc:.4f} yc={yc:.4f} w={w:.4f} h={h:.4f}')

# Data leakage check (strip _aug_N suffixes)
import re
train_originals = set()
for f in train_imgs:
    base = os.path.splitext(f)[0]
    base = re.sub(r'_aug_\d+$', '', base)
    train_originals.add(base)
val_bases = set(os.path.splitext(f)[0] for f in val_imgs)
overlap = train_originals & val_bases
print('[LEAKAGE] Overlap between train originals and val:', overlap if overlap else 'None — clean split')

print("\n" + "="*60)
print("STEP 5: INFERENCE AT MULTIPLE CONFIDENCE THRESHOLDS")
print("="*60)
counts = {0.001:0, 0.05:0, 0.10:0, 0.20:0, 0.30:0, 0.40:0, 0.50:0}
for img_name in val_imgs:
    img = cv2.imread(os.path.join(val_img_dir, img_name))
    for thresh in counts.keys():
        res = model(img, conf=thresh, verbose=False)
        n = len(res[0].boxes)
        counts[thresh] += n
        if thresh <= 0.10:
            for b in res[0].boxes:
                cls = int(b.cls[0])
                c = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                print(f'  [{img_name}] conf={thresh} → {model.names[cls]} @ {c:.4f}  box=[{x1},{y1},{x2},{y2}]')

print()
print('[SUMMARY] Total detections across 4 val images:')
for thresh, n in counts.items():
    verdict = 'OK' if n >= 1 else 'NO DETECTION'
    print(f'  conf={thresh}: {n} detections  [{verdict}]')

print("\n" + "="*60)
print("STEP 6: ROOT CAUSE ANALYSIS")
print("="*60)
max_conf_val = 0.0
for img_name in val_imgs:
    img = cv2.imread(os.path.join(val_img_dir, img_name))
    res = model(img, conf=0.001, verbose=False)
    for b in res[0].boxes:
        max_conf_val = max(max_conf_val, float(b.conf[0]))

print('Max confidence score on ANY val detection at conf=0.001:', round(max_conf_val, 4))
if max_conf_val < 0.05:
    print()
    print('ROOT CAUSE: The model is severely UNDERFIT.')
    print('  → best.pt was saved from epoch 1 (before the model learned anything meaningful).')
    print('  → Ultralytics fitness = 0.1*mAP50 + 0.9*mAP50-95.')
    print('  → Epoch 1 had higher mAP50-95 (0.765) than later epochs (0.408-0.523).')
    print('  → So epoch 1 was incorrectly selected as "best" by fitness score.')
    print('  → BUT: epoch 1 model has confidence scores of ~0.013 on val images.')
    print('  → Well below any usable threshold (0.05+).')
    print()
    print('REQUIRED ACTION: Retrain with LAST.PT instead of best.pt,')
    print('  OR use last.pt directly which corresponds to epoch 16 (trained model).')
    print()
    last_pt = os.path.join(ROOT, 'runs', 'detect', 'train-2', 'weights', 'last.pt')
    print('Testing last.pt:')
    if os.path.isfile(last_pt):
        last_model = YOLO(last_pt)
        print('  last.pt exists, names:', last_model.names)
        for img_name in val_imgs:
            img = cv2.imread(os.path.join(val_img_dir, img_name))
            for thresh in [0.001, 0.05, 0.10, 0.20, 0.30, 0.40]:
                res = last_model(img, conf=thresh, verbose=False)
                n = len(res[0].boxes)
                max_c = max((float(b.conf[0]) for b in res[0].boxes), default=0.0)
                if n > 0 or thresh == 0.001:
                    print(f'  [{img_name}] conf={thresh}: {n} det  max_conf={max_c:.4f}')
    else:
        print('  last.pt not found')
elif max_conf_val < 0.30:
    print('ROOT CAUSE: Model learned partially but confidence is low.')
    print('  → Need more epochs or more training data.')
else:
    print('Model confidence looks good. Issue must be in webcam pipeline.')
