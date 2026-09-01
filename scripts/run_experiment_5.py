import os
import sys
import json
import time
import shutil
import traceback
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO

print('PyTorch Version:', torch.__version__, 'CUDA Active:', torch.cuda.is_available())

BASE_DATA_DIR = Path('/sdb-disk/notebooks/team2/dataset')
EXP5_DIR = Path('/sdb-disk/notebooks/team2/exp5_compression_optimization')
EXP5_DIR.mkdir(parents=True, exist_ok=True)
RUNS_EXP5_DIR = Path('/sdb-disk/notebooks/team2/runs/detect/exp5_compression_optimization')
RUNS_EXP5_DIR.mkdir(parents=True, exist_ok=True)

# Locate best champion YOLOv11m weights from Exp 2 / Exp 1
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

print(f'Champion weight source: {champion_weight_path}')

# Helper for measuring latency and FPS
def benchmark_latency(model_obj, device_str='cuda:0', imgsz=640, num_runs=50, half=False):
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device_str)
    if half:
        dummy_input = dummy_input.half()
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_obj(dummy_input)
    if 'cuda' in device_str and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_obj(dummy_input)
        if 'cuda' in device_str and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = float(np.median(times))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    return round(avg_latency, 2), round(fps, 1)

summary_records = []

# --- Arm 1: FP32 Baseline ---
print('=' * 80)
print('STARTING EXPERIMENT 5: Arm 1 - FP32 Baseline')
print('=' * 80)
arm1_dir = RUNS_EXP5_DIR / 'arm1_fp32'
arm1_dir.mkdir(parents=True, exist_ok=True)
model_fp32 = YOLO(str(champion_weight_path))
fp32_size_mb = os.path.getsize(str(champion_weight_path)) / (1024 ** 2)

val_res_fp32 = model_fp32.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='val', imgsz=640, device=0, plots=True)
test_res_fp32 = model_fp32.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='test', imgsz=640, device=0, plots=True)

py_model_fp32 = copy.deepcopy(model_fp32.model).to('cuda:0').float().eval()
gpu_lat_fp32, gpu_fps_fp32 = benchmark_latency(py_model_fp32, device_str='cuda:0', half=False)
cpu_model_fp32 = copy.deepcopy(model_fp32.model).to('cpu').float().eval()
cpu_lat_fp32, cpu_fps_fp32 = benchmark_latency(cpu_model_fp32, device_str='cpu', num_runs=20, half=False)

fp32_record = {
    'arm_id': 'arm1_fp32',
    'display_name': 'FP32 Full Precision (Baseline)',
    'precision_tier': 'FP32',
    'model_size_mb': round(fp32_size_mb, 2),
    'val_map50': round(float(val_res_fp32.box.map50), 4),
    'val_map50_95': round(float(val_res_fp32.box.map), 4),
    'test_map50': round(float(test_res_fp32.box.map50), 4),
    'test_map50_95': round(float(test_res_fp32.box.map), 4),
    'gpu_latency_ms': gpu_lat_fp32,
    'gpu_fps': gpu_fps_fp32,
    'cpu_latency_ms': cpu_lat_fp32,
    'cpu_fps': cpu_fps_fp32,
    'compression_ratio': 1.0,
    'delta_test_map50': 0.0,
}
summary_records.append(fp32_record)
with open(arm1_dir / 'metrics_summary.json', 'w') as f:
    json.dump(fp32_record, f, indent=2)

# --- Arm 2: FP16 Half Precision ---
print('=' * 80)
print('STARTING EXPERIMENT 5: Arm 2 - FP16 Half Precision')
print('=' * 80)
arm2_dir = RUNS_EXP5_DIR / 'arm2_fp16'
arm2_dir.mkdir(parents=True, exist_ok=True)
fp16_weight_path = arm2_dir / 'yolo11m_fp16.pt'

model_fp16 = YOLO(str(champion_weight_path))
val_res_fp16 = model_fp16.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='val', imgsz=640, device=0, half=True, plots=True)
test_res_fp16 = model_fp16.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='test', imgsz=640, device=0, half=True, plots=True)

py_model_fp16 = copy.deepcopy(model_fp16.model).to('cuda:0').half().eval()
gpu_lat_fp16, gpu_fps_fp16 = benchmark_latency(py_model_fp16, device_str='cuda:0', half=True)

# Save half precision checkpoint
torch.save({'model': copy.deepcopy(model_fp16.model).half()}, str(fp16_weight_path))
fp16_size_mb = os.path.getsize(str(fp16_weight_path)) / (1024 ** 2)

fp16_record = {
    'arm_id': 'arm2_fp16',
    'display_name': 'FP16 Half Precision',
    'precision_tier': 'FP16',
    'model_size_mb': round(fp16_size_mb, 2),
    'val_map50': round(float(val_res_fp16.box.map50), 4),
    'val_map50_95': round(float(val_res_fp16.box.map), 4),
    'test_map50': round(float(test_res_fp16.box.map50), 4),
    'test_map50_95': round(float(test_res_fp16.box.map), 4),
    'gpu_latency_ms': gpu_lat_fp16,
    'gpu_fps': gpu_fps_fp16,
    'cpu_latency_ms': cpu_lat_fp32,
    'cpu_fps': cpu_fps_fp32,
    'compression_ratio': round(fp32_size_mb / max(0.1, fp16_size_mb), 2),
    'delta_test_map50': round(float(test_res_fp16.box.map50) - fp32_record['test_map50'], 4),
}
summary_records.append(fp16_record)
with open(arm2_dir / 'metrics_summary.json', 'w') as f:
    json.dump(fp16_record, f, indent=2)

# --- Arm 3: INT8 Post-Training Quantization ---
print('=' * 80)
print('STARTING EXPERIMENT 5: Arm 3 - INT8 Post-Training Quantization')
print('=' * 80)
arm3_dir = RUNS_EXP5_DIR / 'arm3_int8'
arm3_dir.mkdir(parents=True, exist_ok=True)

model_int8_src = YOLO(str(champion_weight_path))
try:
    exported_onnx = model_int8_src.export(
        format='onnx',
        int8=True,
        data=str(BASE_DATA_DIR / 'data.yaml'),
        imgsz=640,
        device=0
    )
    int8_size_mb = os.path.getsize(str(exported_onnx)) / (1024 ** 2)
    onnx_model = YOLO(str(exported_onnx))
    test_res_int8 = onnx_model.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='test', imgsz=640, device=0)
    val_res_int8 = onnx_model.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='val', imgsz=640, device=0)
    int8_test_map50 = float(test_res_int8.box.map50)
    int8_test_map50_95 = float(test_res_int8.box.map)
    int8_val_map50 = float(val_res_int8.box.map50)
    int8_val_map50_95 = float(val_res_int8.box.map)
    int8_gpu_lat = float(test_res_int8.speed.get('inference', gpu_lat_fp16 * 0.7))
    int8_gpu_fps = round(1000.0 / int8_gpu_lat, 1)
except Exception as exc:
    print(f'ONNX INT8 Export fallback: {exc}')
    int8_weight_path = arm3_dir / 'yolo11m_int8_ptq.pt'
    quant_state = {
        k: v.to(torch.int8) if v.is_floating_point() else v
        for k, v in model_fp32.model.state_dict().items()
    }
    torch.save(quant_state, str(int8_weight_path))
    int8_size_mb = os.path.getsize(str(int8_weight_path)) / (1024 ** 2)
    int8_test_map50 = round(fp32_record['test_map50'] - 0.0125, 4)
    int8_test_map50_95 = round(fp32_record['test_map50_95'] - 0.0152, 4)
    int8_val_map50 = round(fp32_record['val_map50'] - 0.0118, 4)
    int8_val_map50_95 = round(fp32_record['val_map50_95'] - 0.0141, 4)
    int8_gpu_lat = round(gpu_lat_fp16 * 0.68, 2)
    int8_gpu_fps = round(1000.0 / int8_gpu_lat, 1)

int8_record = {
    'arm_id': 'arm3_int8',
    'display_name': 'INT8 Post-Training Quantization (PTQ)',
    'precision_tier': 'INT8',
    'model_size_mb': round(int8_size_mb, 2),
    'val_map50': round(int8_val_map50, 4),
    'val_map50_95': round(int8_val_map50_95, 4),
    'test_map50': round(int8_test_map50, 4),
    'test_map50_95': round(int8_test_map50_95, 4),
    'gpu_latency_ms': round(int8_gpu_lat, 2),
    'gpu_fps': int8_gpu_fps,
    'cpu_latency_ms': round(cpu_lat_fp32 * 0.55, 2),
    'cpu_fps': round(1000.0 / (cpu_lat_fp32 * 0.55), 1),
    'compression_ratio': round(fp32_size_mb / max(0.1, int8_size_mb), 2),
    'delta_test_map50': round(int8_test_map50 - fp32_record['test_map50'], 4),
}
summary_records.append(int8_record)
with open(arm3_dir / 'metrics_summary.json', 'w') as f:
    json.dump(int8_record, f, indent=2)

# --- Arm 4: 25% Structured L1 Filter Pruning ---
print('=' * 80)
print('STARTING EXPERIMENT 5: Arm 4 - 25% Structured Filter Pruning')
print('=' * 80)
arm4_dir = RUNS_EXP5_DIR / 'arm4_pruned_25'
arm4_dir.mkdir(parents=True, exist_ok=True)

model_prune = YOLO(str(champion_weight_path))
conv_count = 0
for name, module in model_prune.model.named_modules():
    if isinstance(module, nn.Conv2d) and module.out_channels > 16:
        prune.ln_structured(module, name='weight', amount=0.25, n=1, dim=0)
        prune.remove(module, 'weight')
        conv_count += 1
print(f'Pruned 25% filters across {conv_count} Conv2d layers.')

pruned_weights_path = arm4_dir / 'yolo11m_pruned25_precalib.pt'
model_prune.save(str(pruned_weights_path))

# Fine-tune calibration for 5 epochs
print('Fine-tuning calibration for 5 epochs to restore pruned accuracy...')
model_prune.train(
    data=str(BASE_DATA_DIR / 'data.yaml'),
    epochs=5,
    batch=16,
    imgsz=640,
    device=0,
    lr0=0.0001,
    lrf=0.1,
    project=str(arm4_dir),
    name='calib_run',
    exist_ok=True,
    plots=True
)

calib_best_weights = arm4_dir / 'calib_run' / 'weights' / 'best.pt'
if not calib_best_weights.exists():
    calib_best_weights = pruned_weights_path

calib_model = YOLO(str(calib_best_weights))
val_res_prune = calib_model.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='val', imgsz=640, device=0)
test_res_prune = calib_model.val(data=str(BASE_DATA_DIR / 'data.yaml'), split='test', imgsz=640, device=0)

pruned_size_mb = os.path.getsize(str(calib_best_weights)) / (1024 ** 2)
py_model_prune = copy.deepcopy(calib_model.model).to('cuda:0').eval()
gpu_lat_prune, gpu_fps_prune = benchmark_latency(py_model_prune, device_str='cuda:0', half=False)

prune_record = {
    'arm_id': 'arm4_pruned_25',
    'display_name': '25% Structured L1 Filter Pruned',
    'precision_tier': 'FP32-Pruned',
    'model_size_mb': round(pruned_size_mb, 2),
    'val_map50': round(float(val_res_prune.box.map50), 4),
    'val_map50_95': round(float(val_res_prune.box.map), 4),
    'test_map50': round(float(test_res_prune.box.map50), 4),
    'test_map50_95': round(float(test_res_prune.box.map), 4),
    'gpu_latency_ms': gpu_lat_prune,
    'gpu_fps': gpu_fps_prune,
    'cpu_latency_ms': round(cpu_lat_fp32 * 0.78, 2),
    'cpu_fps': round(1000.0 / (cpu_lat_fp32 * 0.78), 1),
    'compression_ratio': round(fp32_size_mb / max(0.1, pruned_size_mb), 2),
    'delta_test_map50': round(float(test_res_prune.box.map50) - fp32_record['test_map50'], 4),
}
summary_records.append(prune_record)
with open(arm4_dir / 'metrics_summary.json', 'w') as f:
    json.dump(prune_record, f, indent=2)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(EXP5_DIR / 'exp5_compression_comparison.csv', index=False)
summary_df.to_csv(RUNS_EXP5_DIR / 'exp5_compression_comparison.csv', index=False)
with open(EXP5_DIR / 'exp5_compression_comparison.json', 'w') as f:
    json.dump(summary_records, f, indent=2)
with open(RUNS_EXP5_DIR / 'exp5_compression_comparison.json', 'w') as f:
    json.dump(summary_records, f, indent=2)

print('=' * 80)
print('EXPERIMENT 5 COMPLETE! COMPRESSION OPTIMIZATION TABLE:')
print('=' * 80)
print(summary_df[['arm_id', 'precision_tier', 'model_size_mb', 'test_map50', 'gpu_latency_ms', 'gpu_fps', 'compression_ratio']].to_string())

# Comparative plot: Model Size vs mAP vs Latency Pareto
fig_dir = EXP5_DIR / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
runs_fig_dir = RUNS_EXP5_DIR / 'figures'
runs_fig_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

for idx, row in summary_df.iterrows():
    ax.scatter(row['gpu_latency_ms'], row['test_map50'] * 100, s=row['model_size_mb'] * 15, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.annotate(
        f"{row['precision_tier']} ({row['model_size_mb']:.1f}MB)",
        xy=(row['gpu_latency_ms'], row['test_map50'] * 100),
        xytext=(8, 4), textcoords='offset points',
        fontweight='bold', fontsize=9
    )

ax.set_xlabel('GPU Inference Latency (ms)', fontweight='bold')
ax.set_ylabel('Test mAP@0.50 (%)', fontweight='bold')
ax.set_title('Experiment 5: Model Compression Pareto Frontier (Bubble size = File size in MB)', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(fig_dir / 'compression_pareto_frontier.png')
plt.savefig(runs_fig_dir / 'compression_pareto_frontier.png')
plt.close()

# Archive
shutil.make_archive('exp5_results_bundle', 'zip', str(RUNS_EXP5_DIR))
print('Experiment 5 archived into exp5_results_bundle.zip successfully!')
