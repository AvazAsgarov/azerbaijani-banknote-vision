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
from ultralytics import YOLO

print('PyTorch Version:', torch.__version__, 'CUDA Active:', torch.cuda.is_available())

BASE_DATA_DIR = Path('/sdb-disk/notebooks/team2/dataset')
EXP4_DIR = Path('/sdb-disk/notebooks/team2/exp4_multiscale_resolution')
EXP4_DIR.mkdir(parents=True, exist_ok=True)
RUNS_EXP4_DIR = Path('/sdb-disk/notebooks/team2/runs/detect/exp4_multiscale_resolution')
RUNS_EXP4_DIR.mkdir(parents=True, exist_ok=True)

ARMS = [
    {
        'id': 'res_320',
        'display_name': 'Arm 1: Mobile Edge (320x320)',
        'imgsz': 320,
        'batch': 16,
        'description': 'Ultra-low compute tier for edge deployment.'
    },
    {
        'id': 'res_640',
        'display_name': 'Arm 2: Standard Baseline (640x640)',
        'imgsz': 640,
        'batch': 16,
        'description': 'Industry standard baseline resolution.'
    },
    {
        'id': 'res_1280',
        'display_name': 'Arm 3: High-Fidelity Numismatic (1280x1280)',
        'imgsz': 1280,
        'batch': 8,
        'description': 'High-resolution tier capturing micro-printing and guilloche security patterns.'
    }
]

# Ensure output directory symlinks exist
for arm in ARMS:
    arm_id = arm['id']
    target_link = EXP4_DIR / arm_id
    real_target = RUNS_EXP4_DIR / arm_id
    real_target.mkdir(parents=True, exist_ok=True)
    if not target_link.exists() and not target_link.is_symlink():
        try:
            target_link.symlink_to(real_target)
        except Exception:
            pass

summary_records = []

for arm in ARMS:
    arm_id = arm['id']
    disp_name = arm['display_name']
    img_size = arm['imgsz']
    batch_size = arm['batch']

    print('=' * 80)
    print(f'STARTING EXPERIMENT 4: {disp_name} [imgsz={img_size}, batch={batch_size}]')
    print('=' * 80)

    run_output_dir = RUNS_EXP4_DIR / arm_id
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    try:
        model = YOLO('yolo11m.pt')

        train_results = model.train(
            data=str(BASE_DATA_DIR / 'data.yaml'),
            epochs=50,
            patience=15,
            batch=batch_size,
            imgsz=img_size,
            device=0,
            workers=4,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,
            warmup_epochs=3.0,
            mosaic=1.0,
            mixup=0.15,
            fliplr=0.5,
            degrees=15.0,
            translate=0.15,
            scale=0.5,
            seed=42,
            project=str(RUNS_EXP4_DIR),
            name=arm_id,
            exist_ok=True,
            plots=True,
            save=True,
        )
        train_duration_sec = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        print(f'Training for {arm_id} finished in {train_duration_sec / 60:.2f} minutes. Peak VRAM: {peak_vram_mb:.1f} MB')

        save_dir = Path(train_results.save_dir) if hasattr(train_results, 'save_dir') else run_output_dir
        best_weights = save_dir / 'weights' / 'best.pt'
        if not best_weights.exists():
            best_weights = run_output_dir / 'weights' / 'best.pt'
        if not best_weights.exists():
            best_weights = save_dir / 'weights' / 'last.pt'
        if not best_weights.exists():
            nested_pt = nested_parent / arm_id / 'weights' / 'best.pt'
            if nested_pt.exists():
                best_weights = nested_pt
        print(f"Loaded trained model from: {best_weights}")
        trained_model = YOLO(str(best_weights))

        # Standard Validation set evaluation
        print(f'Evaluating {arm_id} on VALIDATION set at {img_size}x{img_size}...')
        val_metrics = trained_model.val(
            data=str(BASE_DATA_DIR / 'data.yaml'),
            split='val',
            batch=batch_size,
            imgsz=img_size,
            device=0,
            plots=True,
            save_json=True,
        )

        # In-Domain Zero-leakage test set evaluation
        print(f'Evaluating {arm_id} on TEST set at {img_size}x{img_size}...')
        test_metrics = trained_model.val(
            data=str(BASE_DATA_DIR / 'data.yaml'),
            split='test',
            batch=batch_size,
            imgsz=img_size,
            device=0,
            plots=True,
            save_json=True,
        )

        val_map50 = float(val_metrics.box.map50)
        val_map50_95 = float(val_metrics.box.map)
        test_map50 = float(test_metrics.box.map50)
        test_map50_95 = float(test_metrics.box.map)

        val_speed = val_metrics.speed
        latency_ms = val_speed.get('inference', 0.0)
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        class_names = list(val_metrics.names.values())
        per_class_test_ap = {}
        for i, c_name in enumerate(class_names):
            ap50 = float(test_metrics.box.maps[i]) if i < len(test_metrics.box.maps) else 0.0
            per_class_test_ap[c_name] = {
                'ap50': ap50,
                'ap50_95': float(test_metrics.box.all_ap[i].mean()) if hasattr(test_metrics.box, 'all_ap') else ap50
            }

        # Qualitative predictions on fixed test images
        qual_dir = run_output_dir / 'qualitative_samples'
        qual_dir.mkdir(parents=True, exist_ok=True)
        test_img_dir = BASE_DATA_DIR / 'images' / 'test'
        sample_images = sorted(list(test_img_dir.glob('*.jpg')))[:12]
        if sample_images:
            trained_model.predict(
                source=[str(p) for p in sample_images],
                save=True,
                project=str(qual_dir),
                name='predictions',
                exist_ok=True,
                conf=0.25,
                iou=0.45,
                imgsz=img_size,
                device=0,
            )

        record = {
            'arm_id': arm_id,
            'display_name': disp_name,
            'resolution': img_size,
            'batch_size': batch_size,
            'status': 'SUCCESS',
            'train_time_minutes': round(train_duration_sec / 60, 2),
            'peak_vram_mb': round(peak_vram_mb, 1),
            'val_map50': round(val_map50, 4),
            'val_map50_95': round(val_map50_95, 4),
            'test_map50': round(test_map50, 4),
            'test_map50_95': round(test_map50_95, 4),
            'latency_gpu_ms': round(float(latency_ms), 2),
            'fps_gpu': round(float(fps), 1),
            'per_class_test_ap': per_class_test_ap,
        }
        summary_records.append(record)

        with open(run_output_dir / 'metrics_summary.json', 'w') as f:
            json.dump(record, f, indent=2)

    except Exception as exc:
        print(f'ERROR during {arm_id}: {exc}')
        traceback.print_exc()
        torch.cuda.empty_cache()
        summary_records.append({
            'arm_id': arm_id,
            'display_name': disp_name,
            'resolution': img_size,
            'batch_size': batch_size,
            'status': f'FAILED: {exc}',
        })

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(EXP4_DIR / 'exp4_resolution_comparison.csv', index=False)
summary_df.to_csv(RUNS_EXP4_DIR / 'exp4_resolution_comparison.csv', index=False)
with open(EXP4_DIR / 'exp4_resolution_comparison.json', 'w') as f:
    json.dump(summary_records, f, indent=2)
with open(RUNS_EXP4_DIR / 'exp4_resolution_comparison.json', 'w') as f:
    json.dump(summary_records, f, indent=2)

print('=' * 80)
print('EXPERIMENT 4 COMPLETE! MULTISCALE RESOLUTION TABLE:')
print('=' * 80)
print(summary_df[['arm_id', 'resolution', 'test_map50', 'test_map50_95', 'latency_gpu_ms', 'fps_gpu', 'peak_vram_mb']].to_string())

# Comparative plot: Dual-axis Accuracy vs Latency
success_df = summary_df[summary_df['status'] == 'SUCCESS'].copy()
if not success_df.empty:
    fig_dir = EXP4_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    res_values = [str(r) for r in success_df['resolution']]
    x = np.arange(len(res_values))
    width = 0.35

    color_map = '#58a6ff'
    color_lat = '#f0883e'

    bars = ax1.bar(x, success_df['test_map50'] * 100, width, label='Test mAP@0.50 (%)', color=color_map, alpha=0.85)
    ax1.set_xlabel('Input Resolution (Pixels)', fontweight='bold')
    ax1.set_ylabel('Test mAP@0.50 (%)', color=color_map, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_map)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r}x{r}' for r in res_values])
    ax1.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, success_df['latency_gpu_ms'], color=color_lat, marker='o', linewidth=2.5, markersize=8, label='Latency (ms)')
    ax2.set_ylabel('Inference Latency (ms)', color=color_lat, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_lat)

    for xi, yi in zip(x, success_df['latency_gpu_ms']):
        ax2.annotate(f'{yi:.1f} ms',
                     xy=(xi, yi),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, color=color_lat, fontweight='bold')

    plt.title('Experiment 4: Resolution Scaling Dynamics (Test mAP@0.50 vs A100 Latency)', fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / 'resolution_vs_map_latency.png')
    plt.close()

# Archive
shutil.make_archive('exp4_results_bundle', 'zip', str(RUNS_EXP4_DIR))
print('Experiment 4 archived into exp4_results_bundle.zip successfully!')
