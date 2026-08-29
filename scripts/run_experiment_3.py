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

print("PyTorch Version:", torch.__version__, "CUDA Active:", torch.cuda.is_available())

BASE_DATA_DIR = Path('/sdb-disk/notebooks/team2/dataset')
EXP3_DIR = Path('/sdb-disk/notebooks/team2/exp3_color_space_shortcut')
EXP3_DIR.mkdir(parents=True, exist_ok=True)
RUNS_EXP3_DIR = Path('/sdb-disk/notebooks/team2/runs/detect/exp3_color_space_shortcut')
RUNS_EXP3_DIR.mkdir(parents=True, exist_ok=True)

# 1. Dataset Generation: Grayscale and HSV
def prepare_color_variants():
    splits = ['train', 'val', 'test']
    variants = {
        'grayscale': lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        'hsv': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    }

    for var_name, transform_fn in variants.items():
        var_dir = Path(f'/sdb-disk/notebooks/team2/dataset_{var_name}')
        var_images_dir = var_dir / 'images'
        var_labels_dir = var_dir / 'labels'
        var_images_dir.mkdir(parents=True, exist_ok=True)
        var_labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"Checking dataset variant: {var_name}...")
        for split in splits:
            src_split_img = BASE_DATA_DIR / 'images' / split
            dst_split_img = var_images_dir / split
            dst_split_img.mkdir(parents=True, exist_ok=True)

            # Link labels directly
            dst_split_lbl = var_labels_dir / split
            src_split_lbl = BASE_DATA_DIR / 'labels' / split
            if not dst_split_lbl.exists() and not dst_split_lbl.is_symlink():
                dst_split_lbl.symlink_to(src_split_lbl)

            img_files = sorted(list(src_split_img.glob('*.jpg')))
            existing_files = list(dst_split_img.glob('*.jpg'))
            if len(existing_files) != len(img_files):
                print(f"  Generating {len(img_files)} images for {var_name}/{split}...")
                for img_p in img_files:
                    target_p = dst_split_img / img_p.name
                    if not target_p.exists():
                        img = cv2.imread(str(img_p))
                        if img is not None:
                            transformed = transform_fn(img)
                            cv2.imwrite(str(target_p), transformed, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write data.yaml
        yaml_content = f"""path: {var_dir}
train: images/train
val: images/val
test: images/test

names:
  0: '001_azn'
  1: '005_azn'
  2: '010_azn'
  3: '020_azn'
  4: '050_azn'
  5: '100_azn'
  6: '200_azn'
"""
        (var_dir / 'data.yaml').write_text(yaml_content, encoding='utf-8')
        print(f"Variant {var_name} prepared successfully.")

prepare_color_variants()

# 2. Setup Experiment 3 Arms
ARMS = [
    {
        'id': 'arm1_rgb_full',
        'display_name': 'Arm 1: Full Color (RGB Baseline)',
        'data_yaml': str(BASE_DATA_DIR / 'data.yaml'),
        'description': 'Standard 3-channel RGB color space.'
    },
    {
        'id': 'arm2_hsv_space',
        'display_name': 'Arm 2: Decoupled Chromaticity (HSV Color Space)',
        'data_yaml': '/sdb-disk/notebooks/team2/dataset_hsv/data.yaml',
        'description': 'Hue, Saturation, Value decoupled representation.'
    },
    {
        'id': 'arm3_grayscale_shortcut',
        'display_name': 'Arm 3: Monochromatic Shortcut Test (Grayscale)',
        'data_yaml': '/sdb-disk/notebooks/team2/dataset_grayscale/data.yaml',
        'description': 'Zero chromatic information. Forces reliance on numismatic geometry and micro-patterns.'
    }
]

# Ensure output directory symlinks exist
for arm in ARMS:
    arm_id = arm['id']
    target_link = EXP3_DIR / arm_id
    real_target = RUNS_EXP3_DIR / arm_id
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
    data_yaml = arm['data_yaml']

    print('=' * 80)
    print(f'STARTING EXPERIMENT 3: {disp_name}')
    print(f'Data YAML: {data_yaml}')
    print('=' * 80)

    run_output_dir = RUNS_EXP3_DIR / arm_id
    t0 = time.time()

    try:
        model = YOLO('yolo11m.pt')

        train_results = model.train(
            data=data_yaml,
            epochs=50,
            patience=15,
            batch=16,
            imgsz=640,
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
            project=str(RUNS_EXP3_DIR),
            name=arm_id,
            exist_ok=True,
            plots=True,
            save=True,
        )
        train_duration_sec = time.time() - t0
        print(f'Training for {arm_id} finished in {train_duration_sec / 60:.2f} minutes.')

        best_weights = run_output_dir / 'weights' / 'best.pt'
        if not best_weights.exists():
            best_weights = run_output_dir / 'weights' / 'last.pt'

        trained_model = YOLO(str(best_weights))

        # Standard Validation set evaluation
        print(f'Evaluating {arm_id} on VALIDATION set...')
        val_metrics = trained_model.val(
            data=data_yaml,
            split='val',
            batch=16,
            imgsz=640,
            device=0,
            plots=True,
            save_json=True,
        )

        # In-Domain Zero-leakage test set evaluation
        print(f'Evaluating {arm_id} on IN-DOMAIN TEST set...')
        test_metrics = trained_model.val(
            data=data_yaml,
            split='test',
            batch=16,
            imgsz=640,
            device=0,
            plots=True,
            save_json=True,
        )

        # Cross-Domain Shortcut Test: Evaluate on Grayscale test set specifically
        print(f'Evaluating {arm_id} on MONOCHROME SHORTCUT TEST SET...')
        mono_test_metrics = trained_model.val(
            data='/sdb-disk/notebooks/team2/dataset_grayscale/data.yaml',
            split='test',
            batch=16,
            imgsz=640,
            device=0,
            plots=False,
            save_json=False,
        )

        val_map50 = float(val_metrics.box.map50)
        val_map50_95 = float(val_metrics.box.map)
        test_map50 = float(test_metrics.box.map50)
        test_map50_95 = float(test_metrics.box.map)
        mono_test_map50 = float(mono_test_metrics.box.map50)
        mono_test_map50_95 = float(mono_test_metrics.box.map)

        # Shortcut Reliance Ratio calculation
        shortcut_reliance = round(1.0 - (mono_test_map50 / max(0.001, test_map50)), 4)

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
        test_img_dir = Path(data_yaml).parent / 'images' / 'test'
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
                imgsz=640,
                device=0,
            )

        record = {
            'arm_id': arm_id,
            'display_name': disp_name,
            'status': 'SUCCESS',
            'train_time_minutes': round(train_duration_sec / 60, 2),
            'val_map50': round(val_map50, 4),
            'val_map50_95': round(val_map50_95, 4),
            'test_map50': round(test_map50, 4),
            'test_map50_95': round(test_map50_95, 4),
            'mono_test_map50': round(mono_test_map50, 4),
            'mono_test_map50_95': round(mono_test_map50_95, 4),
            'shortcut_reliance_score': shortcut_reliance,
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
            'status': f'FAILED: {exc}',
        })

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(EXP3_DIR / 'exp3_color_space_comparison.csv', index=False)
with open(EXP3_DIR / 'exp3_color_space_comparison.json', 'w') as f:
    json.dump(summary_records, f, indent=2)

print('=' * 80)
print('EXPERIMENT 3 COMPLETE! COLOR SPACE SHORTCUT TABLE:')
print('=' * 80)
print(summary_df[['arm_id', 'test_map50', 'mono_test_map50', 'shortcut_reliance_score']].to_string())

# Comparative plot
success_df = summary_df[summary_df['status'] == 'SUCCESS'].copy()
if not success_df.empty:
    fig_dir = EXP3_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(success_df))
    width = 0.35

    ax.bar(x - width/2, success_df['test_map50'] * 100, width, label='In-Domain Test mAP@0.50', color='#58a6ff')
    ax.bar(x + width/2, success_df['mono_test_map50'] * 100, width, label='Monochrome Shortcut Test mAP@0.50', color='#f0883e')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Experiment 3: Color Space & Shortcut Learning Resilience')
    ax.set_xticks(x)
    clean_labels = [r.replace('Arm ', '').split('(')[0].strip() for r in success_df['display_name']]
    ax.set_xticklabels(clean_labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / 'color_space_shortcut_delta.png')
    plt.close()

# Archive
shutil.make_archive('exp3_results_bundle', 'zip', str(RUNS_EXP3_DIR))
print('Experiment 3 archived into exp3_results_bundle.zip successfully!')
