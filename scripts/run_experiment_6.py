import os
import sys
import json
import time
import shutil
import traceback
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from ultralytics import YOLO

print('PyTorch Version:', torch.__version__, 'CUDA Active:', torch.cuda.is_available())

BASE_DATA_DIR = Path('/sdb-disk/notebooks/team2/dataset')
EXP6_DIR = Path('/sdb-disk/notebooks/team2/exp6_explainable_ai')
EXP6_DIR.mkdir(parents=True, exist_ok=True)
RUNS_EXP6_DIR = Path('/sdb-disk/notebooks/team2/runs/detect/exp6_explainable_ai')
RUNS_EXP6_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR = RUNS_EXP6_DIR / 'heatmaps'
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# Locate champion weights
candidate_weights = [
    Path('/sdb-disk/notebooks/team2/runs/detect/runs/detect/exp2_data_augmentation/arm4_full_composite/weights/best.pt'),
    Path('/sdb-disk/notebooks/team2/runs/detect/exp2_data_augmentation/arm4_full_composite/weights/best.pt'),
    Path('/sdb-disk/notebooks/team2/runs/detect/exp1_architecture_battle/yolo11m/weights/best.pt'),
    Path('yolo11m.pt'),
]

champion_weight_path = None
for cw in candidate_weights:
    if cw.exists():
        champion_weight_path = cw
        break

print(f'Loading champion model for XAI from: {champion_weight_path}')
model = YOLO(str(champion_weight_path))
py_model = model.model.eval().to('cuda:0')

# Extract activations using forward hook
activations = {}
def get_activation(name):
    def hook(m, input, output):
        activations[name] = output.detach()
    return hook

# Hook C2PSA attention block or deep backbone feature layer
target_layer_name = None
for name, module in py_model.named_modules():
    if 'c2psa' in name.lower():
        target_layer_name = name
        break

if target_layer_name is None:
    for name, module in py_model.named_modules():
        if 'sppf' in name.lower():
            target_layer_name = name
            break

if target_layer_name is None:
    target_layer_name = 'model.9'

for name, module in py_model.named_modules():
    if name == target_layer_name:
        module.register_forward_hook(get_activation('target_layer'))
        print(f'Registered forward hook on target layer: {name}')
        break

# Compute EigenCAM representation from feature activation tensor
def compute_eigencam(feature_tensor):
    b, c, h, w = feature_tensor.shape
    reshaped = feature_tensor.squeeze(0).reshape(c, -1).transpose(0, 1)
    reshaped_centered = reshaped - reshaped.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(reshaped_centered, q=1)
    cam = torch.matmul(reshaped_centered, V[:, :1])
    cam = cam.reshape(h, w).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam

# Calculate Numismatic Alignment Score and Normalized Energy Density
def calculate_alignment_metrics(cam_resized, bboxes_pixels):
    total_energy = float(np.sum(cam_resized))
    total_pixels = float(cam_resized.shape[0] * cam_resized.shape[1])
    if total_energy <= 0.0001 or total_pixels <= 0:
        return 0.0, 0.0

    mask = np.zeros_like(cam_resized, dtype=np.float32)
    for (x1, y1, x2, y2) in bboxes_pixels:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(cam_resized.shape[1], int(x2)), min(cam_resized.shape[0], int(y2))
        mask[y1:y2, x1:x2] = 1.0

    box_pixels = float(np.sum(mask))
    if box_pixels <= 0.0001:
        return 0.0, 0.0

    inside_energy = float(np.sum(cam_resized * mask))
    raw_alignment = inside_energy / total_energy
    normalized_density = (inside_energy / box_pixels) / (total_energy / total_pixels)
    return round(raw_alignment, 4), round(normalized_density, 4)

# Parse test images and ground truth labels
test_img_dir = BASE_DATA_DIR / 'images' / 'test'
test_lbl_dir = BASE_DATA_DIR / 'labels' / 'test'

all_test_images = sorted(list(test_img_dir.glob('*.jpg')))
print(f'Total test images available: {len(all_test_images)}')

class_names = {
    0: '001_azn', 1: '005_azn', 2: '010_azn', 3: '020_azn',
    4: '050_azn', 5: '100_azn', 6: '200_azn'
}

# Group test images by denomination for balanced multi-class evaluation
denom_buckets = {c: [] for c in class_names.values()}
for img_p in all_test_images:
    lbl_p = test_lbl_dir / f'{img_p.stem}.txt'
    if lbl_p.exists() and lbl_p.stat().st_size > 0:
        with open(lbl_p) as lf:
            line = lf.readline().strip()
            if line:
                cid = int(line.split()[0])
                cname = class_names.get(cid)
                if cname in denom_buckets:
                    denom_buckets[cname].append(img_p)

balanced_test_images = []
for cname in sorted(denom_buckets.keys()):
    imgs = denom_buckets[cname]
    picked = imgs[:25]
    balanced_test_images.extend(picked)
    print(f'Denomination {cname}: selected {len(picked)} samples for XAI.')

print(f'Total balanced test images selected across all 7 denominations: {len(balanced_test_images)}')

records = []
analyzed_count = 0

for img_p in balanced_test_images:
    lbl_p = test_lbl_dir / f'{img_p.stem}.txt'
    if not lbl_p.exists():
        continue

    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        continue
    h_orig, w_orig = img_bgr.shape[:2]

    # Read ground truth boxes
    gt_boxes = []
    gt_classes = []
    with open(lbl_p, 'r') as lf:
        for line in lf:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * w_orig
                y1 = (cy - bh / 2.0) * h_orig
                x2 = (cx + bw / 2.0) * w_orig
                y2 = (cy + bh / 2.0) * h_orig
                gt_boxes.append((x1, y1, x2, y2))
                gt_classes.append(cid)

    # Preprocess for PyTorch model
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    tensor_in = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to('cuda:0') / 255.0

    # Forward pass
    with torch.no_grad():
        preds = py_model(tensor_in)

    if 'target_layer' not in activations:
        continue

    feat = activations['target_layer']
    if isinstance(feat, (list, tuple)):
        feat = feat[0]

    cam = compute_eigencam(feat)
    cam_full = cv2.resize(cam, (w_orig, h_orig))

    alignment_score, energy_density = calculate_alignment_metrics(cam_full, gt_boxes)

    # Check top prediction
    pred_res = model.predict(source=str(img_p), conf=0.25, iou=0.45, device=0, verbose=False)[0]
    top_pred_cls = int(pred_res.boxes.cls[0].item()) if len(pred_res.boxes) > 0 else -1
    top_pred_conf = float(pred_res.boxes.conf[0].item()) if len(pred_res.boxes) > 0 else 0.0

    denomination = class_names.get(gt_classes[0], 'unknown') if gt_classes else 'unknown'

    rec = {
        'image_name': img_p.name,
        'denomination': denomination,
        'ground_truth_cls': gt_classes[0] if gt_classes else -1,
        'predicted_cls': top_pred_cls,
        'confidence': round(top_pred_conf, 3),
        'correct': bool(gt_classes and top_pred_cls == gt_classes[0]),
        'numismatic_alignment_score': alignment_score,
        'normalized_energy_density': energy_density,
    }
    records.append(rec)

    # Save heatmap overlay across each denomination
    if analyzed_count % 3 == 0 or denomination in ['020_azn', '200_azn']:
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

        for (x1, y1, x2, y2) in gt_boxes:
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(
            overlay,
            f"{denomination} | Alignment: {alignment_score:.1%} | Conf: {top_pred_conf:.2f}",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        out_p = HEATMAP_DIR / f'cam_{img_p.stem}.jpg'
        cv2.imwrite(str(out_p), overlay)

    analyzed_count += 1

xai_df = pd.DataFrame(records)
xai_df.to_csv(EXP6_DIR / 'exp6_xai_summary.csv', index=False)
xai_df.to_csv(RUNS_EXP6_DIR / 'exp6_xai_summary.csv', index=False)

# Denomination-level alignment summary
denom_summary = xai_df.groupby('denomination').agg(
    mean_alignment=('numismatic_alignment_score', 'mean'),
    accuracy=('correct', 'mean'),
    mean_confidence=('confidence', 'mean'),
    sample_count=('image_name', 'count')
).reset_index()

overall_alignment = float(xai_df['numismatic_alignment_score'].mean())
overall_accuracy = float(xai_df['correct'].mean())

summary_dict = {
    'overall_mean_alignment_score': round(overall_alignment, 4),
    'overall_accuracy': round(overall_accuracy, 4),
    'denomination_metrics': denom_summary.to_dict(orient='records'),
    'hard_pair_analysis': {
        '020_azn_alignment': float(denom_summary.loc[denom_summary['denomination'] == '020_azn', 'mean_alignment'].iloc[0]) if '020_azn' in denom_summary['denomination'].values else 0.0,
        '200_azn_alignment': float(denom_summary.loc[denom_summary['denomination'] == '200_azn', 'mean_alignment'].iloc[0]) if '200_azn' in denom_summary['denomination'].values else 0.0,
    }
}

with open(EXP6_DIR / 'exp6_xai_summary.json', 'w') as f:
    json.dump(summary_dict, f, indent=2)
with open(RUNS_EXP6_DIR / 'exp6_xai_summary.json', 'w') as f:
    json.dump(summary_dict, f, indent=2)

print('=' * 80)
print('EXPERIMENT 6 COMPLETE! NUMISMATIC ALIGNMENT SUMMARY:')
print('=' * 80)
print(denom_summary.to_string(index=False))
print(f'\nOverall Numismatic Alignment Score: {overall_alignment:.2%}')

# Plot: Numismatic Alignment by Denomination
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
colors = ['#58a6ff' if d not in ['020_azn', '200_azn'] else '#f85149' for d in denom_summary['denomination']]
bars = ax.bar(denom_summary['denomination'], denom_summary['mean_alignment'] * 100, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Numismatic Alignment Score (%)', fontweight='bold')
ax.set_xlabel('Banknote Denomination', fontweight='bold')
ax.set_title('Experiment 6: Attention Energy Concentration on Banknote Area (EigenCAM)', fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(EXP6_DIR / 'numismatic_alignment_by_denomination.png')
plt.savefig(RUNS_EXP6_DIR / 'numismatic_alignment_by_denomination.png')
plt.close()

# Archive
shutil.make_archive('exp6_results_bundle', 'zip', str(RUNS_EXP6_DIR))
print('Experiment 6 archived into exp6_results_bundle.zip successfully!')
