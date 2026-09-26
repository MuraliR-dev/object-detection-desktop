import os

# ===== Check what train-2 actually ran =====
train2_dir = r'runs/detect/train-2'
args_file = os.path.join(train2_dir, 'args.yaml')
results_file = os.path.join(train2_dir, 'results.csv')

print('[TRAIN-2 CONFIG]')
try:
    import yaml
    with open(args_file) as f:
        args = yaml.safe_load(f)
    for k in ['model','data','epochs','imgsz','batch','workers','patience','device','optimizer']:
        print('  %s: %s' % (k, args.get(k, 'N/A')))
except Exception as e:
    print('  Error reading args.yaml:', e)

print()
print('[TRAIN-2 RESULTS - first 3 and last 5 epochs]')
try:
    with open(results_file) as f:
        lines = f.readlines()
    print('  Header:', lines[0].strip())
    for line in lines[1:4]:
        print(' ', line.strip())
    print('  ...')
    for line in lines[-5:]:
        print(' ', line.strip())
except Exception as e:
    print('  Error reading results.csv:', e)
