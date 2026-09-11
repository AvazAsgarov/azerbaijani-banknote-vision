import os
import sys
import json
import time
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO, RTDETR
import torch

print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

DATA_YAML = 'dataset/data.yaml'
EXP_DIR = Path('exp1_architecture_battle')
EXP_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        'id': 'yolov8m',
        'display_name': 'YOLOv8m (Anchor-Free CNN)',
        'weights': 'yolov8m.pt',
        'cls': YOLO,
    },
    {
        'id': 'yolo11m',
        'display_name': 'YOLOv11m (Spatial Attention CNN)',
        'weights': 'yolo11m.pt',
        'cls': YOLO,
    },
    {
        'id': 'rtdetr_l',
        'display_name': 'RT-DETR-L (Vision Transformer)',
        'weights': 'rtdetr-l.pt',
        'cls': RTDETR,
    },
]

summary_records = []

# DINOv2 baseline entry dynamically loaded from probe report
dino_probe_file = Path('dino_linear_probe_report.json')
dino_data = {}
if dino_probe_file.exists():
    with open(dino_probe_file) as f:
        dino_data = json.load(f)

summary_records.append({
    'model_id': 'dinov2_vitl14',
    'architecture': 'DINOv2 ViT-L/14 (Linear Probe Baseline)',
    'paradigm': 'Frozen Foundation ViT',
    'parameters_m': 304.3,
    'gflops_640': None,
    'val_map50': None,
    'val_map50_95': None,
    'test_map50': None,
    'test_map50_95': None,
    'val_accuracy': round(dino_data.get('validation_metrics', {}).get('accuracy', 0.0), 4) if dino_data else None,
    'test_accuracy': round(dino_data.get('test_metrics', {}).get('accuracy', 0.0), 4) if dino_data else None,
    'val_macro_f1': round(dino_data.get('validation_metrics', {}).get('macro_f1', 0.0), 4) if dino_data else None,
    'test_macro_f1': round(dino_data.get('test_metrics', {}).get('macro_f1', 0.0), 4) if dino_data else None,
    'latency_gpu_ms': None,
    'fps_gpu': None,
})

for m_cfg in MODELS:
    m_id = m_cfg['id']
    disp_name = m_cfg['display_name']
    w_path = m_cfg['weights']
    ModelCls = m_cfg['cls']

    print('=' * 80)
    print(f'STARTING TRAINING FOR: {disp_name} ({w_path})')
    print('=' * 80)

    run_output_dir = EXP_DIR / m_id
    metrics_summary_file = run_output_dir / 'metrics_summary.json'
    last_weights = run_output_dir / 'weights' / 'last.pt'

    if metrics_summary_file.exists():
        print(f'Model {m_id} already has completed evaluation metrics. Skipping retraining.')
        with open(metrics_summary_file) as f:
            summary_records.append(json.load(f))
        continue

    # Setup live telemetry status callback
    import subprocess, psutil
    def make_status_callback(model_id):
        def dump_status(trainer):
            try:
                cur_ep = trainer.epoch + 1
                max_ep = trainer.epochs
                free_b, total_b = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
                gpu_used = round((total_b - free_b) / (1024 * 1024), 1)
                gpu_tot = round(total_b / (1024 * 1024), 1)

                temp_val = None
                try:
                    smi = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                    s = smi.stdout.strip()
                    if s.isdigit():
                        temp_val = float(s)
                except Exception:
                    pass

                ram = psutil.virtual_memory()
                cpu_pct = psutil.cpu_percent()

                st = {
                    "active_model": model_id,
                    "epoch": cur_ep,
                    "max_epochs": max_ep,
                    "pid": os.getpid(),
                    "is_running": True,
                    "timestamp": time.time(),
                    "hardware": {
                        "gpu_memory_used_mb": gpu_used,
                        "gpu_memory_total_mb": gpu_tot,
                        "gpu_temp_c": temp_val,
                        "cpu_util_pct": cpu_pct,
                        "ram_used_mb": round(ram.used / (1024 * 1024), 1),
                        "ram_total_mb": round(ram.total / (1024 * 1024), 1),
                        "is_running": True,
                        "pid": os.getpid(),
                        "timestamp": time.time()
                    }
                }
                with open(EXP_DIR / 'training_status.json', 'w') as sf:
                    json.dump(st, sf, indent=2)
            except Exception as exc:
                print(f"Status dump warning: {exc}")
        return dump_status

    # Initial status
    try:
        free_b, total_b = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        init_st = {
            "active_model": m_id,
            "epoch": 71 if last_weights.exists() else 0,
            "max_epochs": 100,
            "pid": os.getpid(),
            "is_running": True,
            "timestamp": time.time(),
            "hardware": {
                "gpu_memory_used_mb": round((total_b - free_b) / (1024 * 1024), 1),
                "gpu_memory_total_mb": round(total_b / (1024 * 1024), 1),
                "gpu_temp_c": None,
                "cpu_util_pct": None,
                "ram_used_mb": round(psutil.virtual_memory().used / (1024 * 1024), 1),
                "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
                "is_running": True,
                "pid": os.getpid(),
                "timestamp": time.time()
            }
        }
        with open(EXP_DIR / 'training_status.json', 'w') as sf:
            json.dump(init_st, sf, indent=2)
    except Exception:
        pass

    # 1. Train or Resume
    t0 = time.time()
    if last_weights.exists():
        print(f'Found existing checkpoint for {m_id} at {last_weights}. Resuming training...')
        model = ModelCls(str(last_weights))
        model.add_callback("on_train_epoch_end", make_status_callback(m_id))
        train_results = model.train(resume=True)
    else:
        model = ModelCls(w_path)
        model.add_callback("on_train_epoch_end", make_status_callback(m_id))
        train_results = model.train(
            data=DATA_YAML,
            epochs=100,
            patience=20,
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
            mixup=0.0,
            fliplr=0.5,
            seed=42,
            project=str(EXP_DIR),
            name=m_id,
            exist_ok=True,
            plots=True,
            save=True,
        )
    train_duration_sec = time.time() - t0
    print(f'Training for {m_id} finished in {train_duration_sec / 60:.2f} minutes.')

    best_weights = run_output_dir / 'weights' / 'best.pt'
    if not best_weights.exists():
        best_weights = run_output_dir / 'weights' / 'last.pt'
    if not best_weights.exists():
        alt_best = Path('runs/detect') / EXP_DIR / m_id / 'weights' / 'best.pt'
        if alt_best.exists():
            best_weights = alt_best
        else:
            best_weights = Path('runs/detect') / EXP_DIR / m_id / 'weights' / 'last.pt'

    trained_model = ModelCls(str(best_weights))

    # 2. Validation Set Evaluation
    print(f'Evaluating {m_id} on VALIDATION set...')
    val_metrics = trained_model.val(
        data=DATA_YAML,
        split='val',
        batch=16,
        imgsz=640,
        device=0,
        plots=True,
        save_json=True,
    )

    # 3. Test Set Evaluation
    print(f'Evaluating {m_id} on ZERO-LEAKAGE TEST set...')
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
        ap50_val = float(test_metrics.box.ap50[i]) if hasattr(test_metrics.box, 'ap50') and i < len(test_metrics.box.ap50) else 0.0
        ap50_95_val = float(test_metrics.box.maps[i]) if hasattr(test_metrics.box, 'maps') and i < len(test_metrics.box.maps) else 0.0
        per_class_test_ap[c_name] = {
            'ap50': ap50_val,
            'ap50_95': ap50_95_val
        }

    n_params = sum(p.numel() for p in trained_model.model.parameters()) / 1e6

    # Qualitative predictions on test set
    print(f'Generating qualitative samples for {m_id}...')
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
        'model_id': m_id,
        'architecture': disp_name,
        'paradigm': 'Anchor-Free CNN' if 'YOLO' in disp_name else 'Vision Transformer',
        'parameters_m': round(float(n_params), 2),
        'train_time_minutes': round(train_duration_sec / 60, 2),
        'val_map50': round(val_map50, 4),
        'val_map50_95': round(val_map50_95, 4),
        'val_map75': round(val_map75, 4),
        'test_map50': round(test_map50, 4),
        'test_map50_95': round(test_map50_95, 4),
        'test_map75': round(test_map75, 4),
        'latency_gpu_ms': round(float(latency_ms), 2),
        'fps_gpu': round(float(fps), 1),
        'per_class_test_ap': per_class_test_ap,
    }
    summary_records.append(record)
    with open(run_output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(record, f, indent=2)
    alt_sum = Path('runs/detect') / EXP_DIR / m_id / 'metrics_summary.json'
    if alt_sum.parent.exists():
        with open(alt_sum, 'w') as f:
            json.dump(record, f, indent=2)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(EXP_DIR / 'exp1_architecture_comparison.csv', index=False)
with open(EXP_DIR / 'exp1_architecture_comparison.json', 'w') as f:
    json.dump(summary_records, f, indent=2)

# Write final status
with open(EXP_DIR / 'training_status.json', 'w') as sf:
    json.dump({
        'is_running': False,
        'active_model': None,
        'completed': True,
        'timestamp': time.time(),
        'hardware': {'is_running': False, 'pid': None}
    }, sf, indent=2)

print('=' * 80)
print('EXPERIMENT 1 COMPLETE! SUMMARY TABLE:')
print('=' * 80)
print(summary_df[['model_id', 'parameters_m', 'val_map50_95', 'test_map50_95', 'test_map50', 'latency_gpu_ms', 'fps_gpu']].to_string())

# Pareto plot
det_df = summary_df[summary_df['test_map50_95'].notnull()].copy()
if not det_df.empty:
    fig_dir = EXP_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6), dpi=300)
    for idx, (_, row) in enumerate(det_df.iterrows()):
        plt.scatter(
            row['latency_gpu_ms'],
            row['test_map50_95'] * 100,
            s=row['parameters_m'] * 8 + 120,
            alpha=0.85,
            edgecolors='black',
            linewidth=1.5,
            label=row['architecture'] + ' (' + str(row['parameters_m']) + 'M params)'
        )
        label_text = row['model_id'] + ' (' + str(round(row['test_map50_95']*100, 1)) + '%, ' + str(row['latency_gpu_ms']) + 'ms)'
        plt.annotate(
            label_text,
            (row['latency_gpu_ms'] + 0.1, row['test_map50_95'] * 100 - 0.5),
            fontsize=9,
            weight='bold'
        )
    plt.title('Pareto Frontier: A100 Latency vs. Test Detection Accuracy (mAP@0.5:0.95)', fontsize=12, weight='bold')
    plt.xlabel('A100 Latency (ms / image, batch=1)')
    plt.ylabel('Test mAP@0.5:0.95 (%)')
    plt.legend(frameon=True, fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(fig_dir / 'pareto_latency_vs_map.png')
    plt.close()

shutil.make_archive('exp1_results_bundle', 'zip', str(EXP_DIR))
print('Artifacts archived into exp1_results_bundle.zip successfully!')
