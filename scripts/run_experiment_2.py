import os
import sys
import json
import time
import shutil
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from ultralytics import YOLO

print("PyTorch Version:", torch.__version__, "CUDA Active:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

DATA_YAML = 'dataset/data.yaml'
EXP_DIR = Path('exp2_data_augmentation')
EXP_DIR.mkdir(parents=True, exist_ok=True)
RUNS_EXP2_DIR = Path('runs/detect/exp2_data_augmentation')
RUNS_EXP2_DIR.mkdir(parents=True, exist_ok=True)

# Symlink each arm directory to resolve weights cleanly
ARMS = [
    {
        'id': 'arm1_none_raw',
        'display_name': 'Arm 1: Raw Baseline (Zero Augmentation)',
        'hparams': {
            'mosaic': 0.0,
            'mixup': 0.0,
            'fliplr': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
        }
    },
    {
        'id': 'arm2_geometric',
        'display_name': 'Arm 2: Pure Geometric (Rotation, Scale, Shear, Flip)',
        'hparams': {
            'mosaic': 0.0,
            'mixup': 0.0,
            'fliplr': 0.5,
            'degrees': 15.0,
            'translate': 0.15,
            'scale': 0.5,
            'shear': 2.5,
            'perspective': 0.0005,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
        }
    },
    {
        'id': 'arm3_photometric',
        'display_name': 'Arm 3: Pure Photometric (Color Jitter & Lighting)',
        'hparams': {
            'mosaic': 0.0,
            'mixup': 0.0,
            'fliplr': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.025,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }
    },
    {
        'id': 'arm4_full_composite',
        'display_name': 'Arm 4: Full Composite (Mosaic, Mixup, Geometric & Color)',
        'hparams': {
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.1,
            'fliplr': 0.5,
            'degrees': 15.0,
            'translate': 0.15,
            'scale': 0.5,
            'shear': 2.5,
            'perspective': 0.0005,
            'hsv_h': 0.025,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }
    }
]

for arm in ARMS:
    arm_id = arm['id']
    target_link = EXP_DIR / arm_id
    real_target = Path('/sdb-disk/notebooks/team2') / RUNS_EXP2_DIR / arm_id
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
    hparams = arm['hparams']

    print('=' * 80)
    print(f'STARTING EXPERIMENT 2 ABLATION: {disp_name}')
    print('Hyperparameters:', json.dumps(hparams, indent=2))
    print('=' * 80)

    run_output_dir = RUNS_EXP2_DIR / arm_id
    t0 = time.time()

    try:
        model = YOLO('yolo11m.pt')

        train_args = {
            'data': DATA_YAML,
            'epochs': 50,
            'patience': 15,
            'batch': 16,
            'imgsz': 640,
            'device': 0,
            'workers': 4,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'cos_lr': True,
            'warmup_epochs': 3.0,
            'seed': 42,
            'project': str(RUNS_EXP2_DIR),
            'name': arm_id,
            'exist_ok': True,
            'plots': True,
            'save': True,
        }
        train_args.update(hparams)

        train_results = model.train(**train_args)
        train_duration_sec = time.time() - t0
        print(f'Training for {arm_id} finished in {train_duration_sec / 60:.2f} minutes.')

        best_weights = run_output_dir / 'weights' / 'best.pt'
        if not best_weights.exists():
            best_weights = run_output_dir / 'weights' / 'last.pt'

        trained_model = YOLO(str(best_weights))

        # Validation set evaluation
        print(f'Evaluating {arm_id} on VALIDATION set...')
        val_metrics = trained_model.val(
            data=DATA_YAML,
            split='val',
            batch=16,
            imgsz=640,
            device=0,
            plots=True,
            save_json=True,
        )

        # Zero-leakage test set evaluation
        print(f'Evaluating {arm_id} on ZERO-LEAKAGE TEST set...')
        test_metrics = trained_model.val(
            data=DATA_YAML,
            split='test',
            batch=16,
            imgsz=640,
            device=0,
            plots=True,
            save_json=True,
        )

        val_speed = val_metrics.speed
        latency_ms = val_speed.get('inference', 0.0)
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        val_map50 = float(val_metrics.box.map50)
        val_map50_95 = float(val_metrics.box.map)
        val_map75 = float(val_metrics.box.map75)

        test_map50 = float(test_metrics.box.map50)
        test_map50_95 = float(test_metrics.box.map)
        test_map75 = float(test_metrics.box.map75)

        class_names = list(val_metrics.names.values())
        per_class_test_ap = {}
        for i, c_name in enumerate(class_names):
            ap50 = float(test_metrics.box.maps[i]) if i < len(test_metrics.box.maps) else 0.0
            per_class_test_ap[c_name] = {
                'ap50': ap50,
                'ap50_95': float(test_metrics.box.all_ap[i].mean()) if hasattr(test_metrics.box, 'all_ap') else ap50
            }

        # Qualitative predictions on fixed test images
        print(f'Generating qualitative evaluation samples for {arm_id}...')
        qual_dir = run_output_dir / 'qualitative_samples'
        qual_dir.mkdir(parents=True, exist_ok=True)
        test_img_dir = Path('dataset/images/test')
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
            'val_map75': round(val_map75, 4),
            'test_map50': round(test_map50, 4),
            'test_map50_95': round(test_map50_95, 4),
            'test_map75': round(test_map75, 4),
            'generalization_gap_map50': round(val_map50 - test_map50, 4),
            'latency_gpu_ms': round(float(latency_ms), 2),
            'fps_gpu': round(float(fps), 1),
            'per_class_test_ap': per_class_test_ap,
            'hyperparameters': hparams,
        }
        summary_records.append(record)

        with open(run_output_dir / 'metrics_summary.json', 'w') as f:
            json.dump(record, f, indent=2)

    except Exception as exc:
        print(f'ERROR occurred during {arm_id}: {exc}')
        traceback.print_exc()
        torch.cuda.empty_cache()
        summary_records.append({
            'arm_id': arm_id,
            'display_name': disp_name,
            'status': f'FAILED: {exc}',
        })

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(EXP_DIR / 'exp2_augmentation_comparison.csv', index=False)
with open(EXP_DIR / 'exp2_augmentation_comparison.json', 'w') as f:
    json.dump(summary_records, f, indent=2)

print('=' * 80)
print('EXPERIMENT 2 COMPLETE! COMPARISON TABLE:')
print('=' * 80)
print(summary_df[['arm_id', 'val_map50_95', 'test_map50_95', 'test_map50', 'generalization_gap_map50']].to_string())

# Generate comparative figure
success_df = summary_df[summary_df['status'] == 'SUCCESS'].copy()
if not success_df.empty:
    fig_dir = EXP_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(len(success_df))
    width = 0.25

    ax.bar(x - width, success_df['val_map50_95'] * 100, width, label='Val mAP@0.50:0.95', color='#58a6ff')
    ax.bar(x, success_df['test_map50_95'] * 100, width, label='Test mAP@0.50:0.95', color='#3fb950')
    ax.bar(x + width, success_df['test_map50'] * 100, width, label='Test mAP@0.50', color='#e3b341')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Experiment 2: Data Augmentation Ablation Impact on Zero-Leakage Performance')
    ax.set_xticks(x)
    clean_labels = [r.replace('Arm ', '') for r in success_df['display_name']]
    ax.set_xticklabels(clean_labels, fontsize=8, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / 'augmentation_ablation_delta.png')
    plt.close()

# Also copy all summary items into EXP_DIR for archiving
shutil.make_archive('exp2_results_bundle', 'zip', str(RUNS_EXP2_DIR))
print('Experiment 2 archived successfully into exp2_results_bundle.zip!')
