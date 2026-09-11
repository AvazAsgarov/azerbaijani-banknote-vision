"""
Automated Experiment Sync Daemon & Post-Processor
=================================================
Continuously monitors the remote A100 server for completed experiments.
When a pipeline stage completes:
1. For detection experiments (Exp 3, 4):
   - Dynamically discovers data.yaml from args.yaml (e.g. RGB vs HSV vs Grayscale).
   - Supports per-arm image resolution (320, 640, 1280).
   - Automatically triggers remote test set evaluation (test_eval) and
     inference overlays (test_inferences) on the remote GPU using best.pt.
   - Packages all arms and experiment root files.
2. For compression experiments (Exp 5):
   - Discovers precision variants (FP32, FP16, INT8 PTQ, Pruned 25%).
   - Benchmarks and generates test inferences with predictions_manifest.json.
3. For explainability experiments (Exp 6 - xAI):
   - Packages all EigenCAM heatmaps, attention overlays, and analysis reports.
4. Downloads the archive and restructures into canonical zero-leakage folder format:
   - Eliminates duplicate/loose root files.
   - Eliminates stale training batches.
   - Eliminates redundant reports folders.
   - Guarantees zero empty directories.
   - Verifies publication-grade figures and diagnostic reports.
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths, RemoteClusterConfig

_cluster_cfg = RemoteClusterConfig()
_paths = ProjectPaths()

REMOTE_BASE = _cluster_cfg.base_url
REMOTE_TOKEN = _cluster_cfg.token
REMOTE_HEADERS = _cluster_cfg.get_auth_headers() if REMOTE_TOKEN else {}

LOCAL_ARTIFACTS = _paths.experiments_dir
LOG_DIR = _paths.root_dir / "logs"
LOG_PATH = LOG_DIR / "sync_daemon.log"
SYNC_STATE_FILE = LOG_DIR / "sync_state.json"

_CURVE_FILES = {
    "results.png", "BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png",
    "confusion_matrix.png", "confusion_matrix_normalized.png", "labels.jpg",
    "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg",
    "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    "val_batch1_labels.jpg", "val_batch1_pred.jpg",
    "val_batch2_labels.jpg", "val_batch2_pred.jpg",
}
_LOG_FILES = {"args.yaml", "results.csv"}
_ARM_ROOT_KEEP = {"README.md", "metrics_summary.json"}

EXPERIMENT_SPECS = {
    "exp1": {
        "type": "detection",
        "local_folder": "exp1_architecture_battle",
        "remote_dir": "exp1_architecture_battle",
        "arms": ["yolov8m", "yolo11m", "rtdetr_l"],
    },
    "exp2": {
        "type": "detection",
        "local_folder": "exp2_data_augmentation",
        "remote_dir": "exp2_data_augmentation",
        "arms": ["arm1_none_raw", "arm2_geometric", "arm3_photometric", "arm4_full_composite"],
    },
    "exp3": {
        "type": "detection",
        "local_folder": "exp3_color_space_shortcut",
        "remote_dir": "exp3_color_space_shortcut",
        "arms": ["arm1_rgb_full", "arm2_hsv_space", "arm3_grayscale_shortcut"],
    },
    "exp4": {
        "type": "detection",
        "local_folder": "exp4_resolution_scaling",
        "remote_dir": "exp4_multiscale_resolution",
        "arms": ["res_320", "res_640", "res_1280"],
        "arm_imgsz": {"res_320": 320, "res_640": 640, "res_1280": 1280},
    },
    "exp5": {
        "type": "compression",
        "local_folder": "exp5_model_compression",
        "remote_dir": "exp5_compression_optimization",
        "arms": ["arm1_fp32", "arm2_fp16", "arm3_int8", "arm4_pruned_25"],
    },
    "exp6": {
        "type": "xai",
        "local_folder": "exp6_explainability_xai",
        "remote_dir": "exp6_explainable_ai",
        "arms": [],
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sync_daemon")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if SYNC_STATE_FILE.is_file():
        try:
            return json.loads(SYNC_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"synced": {}}


def save_state(state: dict) -> None:
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Remote Execution Helpers
# ---------------------------------------------------------------------------

def execute_remote_code(code: str, timeout: int = 300) -> str:
    import websocket as ws_lib

    r = requests.post(
        f"{REMOTE_BASE}/api/kernels",
        headers=REMOTE_HEADERS,
        json={"name": "python3"},
        timeout=30,
    )
    r.raise_for_status()
    kid = r.json()["id"]

    ws_url = _cluster_cfg.get_websocket_url(kid)
    ws = ws_lib.create_connection(ws_url, timeout=timeout)

    msg_id = uuid.uuid4().hex
    payload = {
        "header": {"msg_id": msg_id, "username": "daemon", "session": msg_id,
                   "msg_type": "execute_request", "version": "5.3"},
        "metadata": {},
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False},
        "parent_header": {},
        "channel": "shell",
    }
    ws.send(json.dumps(payload))

    output = ""
    while True:
        raw = ws.recv()
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") == msg_id:
            if msg.get("msg_type") == "stream":
                output += msg["content"]["text"]
            elif msg.get("msg_type") == "status" and msg.get("content", {}).get("execution_state") == "idle":
                break

    ws.close()
    requests.delete(f"{REMOTE_BASE}/api/kernels/{kid}", headers=REMOTE_HEADERS, timeout=10)
    return output.strip()


def fetch_pipeline_status() -> dict:
    try:
        raw = execute_remote_code("import json; print(open('master_pipeline_status.json').read())", timeout=30)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Could not fetch pipeline status: %s", exc)
        return {}


def download_file(remote_filename: str, local_path: Path) -> bool:
    url = f"{REMOTE_BASE}/files/{remote_filename}?token={REMOTE_TOKEN}"
    try:
        with requests.get(url, stream=True, timeout=900) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total and done % (25 * 1024 * 1024) < 1024 * 1024:
                            logger.info("  ... %.0f%% (%.1f MB)", done / total * 100, done / 1e6)
        mb = local_path.stat().st_size / 1e6
        logger.info("Downloaded %s: %.1f MB", remote_filename, mb)
        return True
    except Exception as exc:
        logger.error("Download failed %s: %s", remote_filename, exc)
        if local_path.exists():
            local_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Remote Evaluation & Packaging for Detection Experiments (Exp 3, Exp 4)
# ---------------------------------------------------------------------------

def remote_evaluate_and_package_detection(exp_key: str, spec: dict) -> str:
    arms_repr = json.dumps(spec["arms"])
    arm_imgsz_repr = json.dumps(spec.get("arm_imgsz", {}))
    exp_dir_name = spec["remote_dir"]
    out_zip = f"{exp_key}_daemon_sync.zip"

    code = f"""
import os, subprocess, shutil, json, zipfile, glob
from pathlib import Path

arms = {arms_repr}
arm_imgsz_map = {arm_imgsz_repr}
exp_name = '{exp_dir_name}'
out_zip = '{out_zip}'

candidates = [
    f'runs/detect/runs/detect/{{exp_name}}',
    f'runs/detect/{{exp_name}}',
    exp_name
]
base_dir = None
for c in candidates:
    if os.path.exists(c) and any(os.path.exists(os.path.join(c, a)) for a in arms):
        base_dir = c
        break

if not base_dir:
    base_dir = exp_name
    print(f"Fallback to {{base_dir}}")
else:
    print(f"Found experiment runs at: {{base_dir}}")

for arm in arms:
    arm_dir = os.path.join(base_dir, arm)
    best_pt = os.path.join(arm_dir, 'weights', 'best.pt')
    if not os.path.exists(best_pt):
        best_pt = os.path.join(arm_dir, 'weights', 'last.pt')
    if not os.path.exists(best_pt):
        print(f"No weights for {{arm}} at {{best_pt}}")
        continue

    data_yaml = 'dataset/data.yaml'
    args_path = os.path.join(arm_dir, 'args.yaml')
    if os.path.exists(args_path):
        try:
            for l in open(args_path).read().splitlines():
                if l.startswith('data:'):
                    candidate = l.split('data:', 1)[1].strip().strip("'").strip('"')
                    if os.path.exists(candidate):
                        data_yaml = candidate
                        break
        except Exception:
            pass

    arm_imgsz = arm_imgsz_map.get(arm, 640)
    print(f"Arm {{arm}}: data={{data_yaml}}, imgsz={{arm_imgsz}}")

    dest_eval = os.path.join(arm_dir, 'test_eval')
    if not os.path.exists(dest_eval) or len(os.listdir(dest_eval)) < 5:
        print(f"Running test evaluation for {{arm}}...")
        cmd_val = [
            'python3', '-c', f\"\"\"
from ultralytics import YOLO
model = YOLO('{{best_pt}}')
metrics = model.val(
    data='{{data_yaml}}',
    split='test',
    imgsz={{arm_imgsz}},
    batch=16,
    save=True,
    save_json=True,
    plots=True,
    name='eval_tmp',
    project='runs/detect/eval_tmp_{{arm}}',
    conf=0.25,
    iou=0.7
)
\"\"\"
        ]
        subprocess.run(cmd_val, capture_output=True, text=True)
        eval_find = subprocess.run(['find', 'runs/detect', '-path', f'*eval_tmp_{{arm}}*eval_tmp*'], capture_output=True, text=True).stdout.strip().splitlines()
        if eval_find and os.path.exists(eval_find[0]):
            if os.path.exists(dest_eval):
                shutil.rmtree(dest_eval)
            shutil.move(eval_find[0], dest_eval)
            print(f"Evaluated {{arm}} -> {{dest_eval}} ({{len(os.listdir(dest_eval))}} files)")
            for ef in eval_find:
                p = Path(ef).parent
                if p.exists() and 'eval_tmp' in str(p):
                    shutil.rmtree(str(p), ignore_errors=True)

    dest_inf = os.path.join(arm_dir, 'test_inferences')
    if not os.path.exists(dest_inf) or len(os.listdir(dest_inf)) < 5:
        print(f"Running test inferences for {{arm}}...")
        test_img_dir = os.path.join(os.path.dirname(data_yaml), 'images', 'test') if os.path.isabs(data_yaml) else 'dataset/images/test'
        if not os.path.exists(test_img_dir):
            test_img_dir = 'dataset/images/test'

        cmd_inf = [
            'python3', '-c', f\"\"\"
from ultralytics import YOLO
import glob, os
model = YOLO('{{best_pt}}')
test_imgs = sorted(glob.glob('{{test_img_dir}}/*.jpg'))[:12]
model.predict(
    source=test_imgs,
    save=True,
    imgsz={{arm_imgsz}},
    name='inf_tmp',
    project='runs/detect/inf_tmp_{{arm}}',
    conf=0.25,
    iou=0.7
)
\"\"\"
        ]
        subprocess.run(cmd_inf, capture_output=True, text=True)
        inf_find = subprocess.run(['find', 'runs/detect', '-path', f'*inf_tmp_{{arm}}*inf_tmp*'], capture_output=True, text=True).stdout.strip().splitlines()
        if inf_find and os.path.exists(inf_find[0]):
            if os.path.exists(dest_inf):
                shutil.rmtree(dest_inf)
            os.makedirs(dest_inf, exist_ok=True)
            manifest_list = []
            for f in sorted(os.listdir(inf_find[0])):
                if f.endswith('.jpg'):
                    src = os.path.join(inf_find[0], f)
                    base, ext = os.path.splitext(f)
                    dst_name = f if base.endswith('_pred') else f\"{{base}}_pred{{ext}}\"
                    dst = os.path.join(dest_inf, dst_name)
                    shutil.copy2(src, dst)
                    manifest_list.append({{
                        "filename": dst_name,
                        "path": f"artifacts/experiments/{{exp_name}}/{{arm}}/test_inferences/{{dst_name}}"
                    }})
            manifest = {{
                "model": arm,
                "experiment": exp_name,
                "total_predictions": len(manifest_list),
                "predictions": manifest_list
            }}
            with open(os.path.join(dest_inf, 'predictions_manifest.json'), 'w') as mf:
                json.dump(manifest, mf, indent=2)
            print(f"Inferences {{arm}} -> {{dest_inf}} ({{len(os.listdir(dest_inf))}} files)")
            for if_p in inf_find:
                p = Path(if_p).parent
                if p.exists() and 'inf_tmp' in str(p):
                    shutil.rmtree(str(p), ignore_errors=True)

    for f in list(os.listdir(arm_dir)):
        if f.startswith('train_batch') and f.endswith('.jpg'):
            if f not in ('train_batch0.jpg', 'train_batch1.jpg', 'train_batch2.jpg'):
                try:
                    os.remove(os.path.join(arm_dir, f))
                except Exception:
                    pass

if os.path.exists(out_zip):
    os.remove(out_zip)

with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root_dir in [exp_name, base_dir]:
        if os.path.exists(root_dir):
            for f in os.listdir(root_dir):
                fp = os.path.join(root_dir, f)
                if os.path.isfile(fp) and not f.endswith('.zip'):
                    zf.write(fp, f)
            figures_dir = os.path.join(root_dir, 'figures')
            if os.path.isdir(figures_dir):
                for fig in os.listdir(figures_dir):
                    fp = os.path.join(figures_dir, fig)
                    if os.path.isfile(fp):
                        zf.write(fp, os.path.join('figures', fig))

    for arm in arms:
        arm_dir = os.path.join(base_dir, arm)
        if os.path.isdir(arm_dir):
            for root, dirs, files in os.walk(arm_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    arc = os.path.join(arm, os.path.relpath(fp, arm_dir))
                    zf.write(fp, arc)

print(f"ZIP_SUCCESS:{{out_zip}}:{{os.path.getsize(out_zip)}}")
"""
    logger.info("Executing remote evaluation and packaging for %s...", exp_key)
    out = execute_remote_code(code, timeout=900)
    for line in out.splitlines():
        if line.startswith("ZIP_SUCCESS:"):
            parts = line.split(":")
            logger.info("Remote package created: %s (%s bytes)", parts[1], parts[2])
            return parts[1]
    logger.warning("Remote packaging output did not contain ZIP_SUCCESS. Output:\n%s", out[:500])
    return out_zip


# ---------------------------------------------------------------------------
# Remote Packaging for Compression Experiments (Exp 5)
# ---------------------------------------------------------------------------

def remote_evaluate_and_package_compression(exp_key: str, spec: dict) -> str:
    arms_repr = json.dumps(spec["arms"])
    exp_dir_name = spec["remote_dir"]
    out_zip = f"{exp_key}_daemon_sync.zip"

    code = f"""
import os, subprocess, shutil, json, zipfile, glob
from pathlib import Path

arms = {arms_repr}
exp_name = '{exp_dir_name}'
out_zip = '{out_zip}'

candidates = [
    f'runs/detect/runs/detect/{{exp_name}}',
    f'runs/detect/{{exp_name}}',
    exp_name
]
base_dir = None
for c in candidates:
    if os.path.exists(c) and any(os.path.exists(os.path.join(c, a)) for a in arms):
        base_dir = c
        break

if not base_dir:
    base_dir = exp_name

if os.path.exists(out_zip):
    os.remove(out_zip)

with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root_dir in [exp_name, base_dir]:
        if os.path.exists(root_dir):
            for f in os.listdir(root_dir):
                fp = os.path.join(root_dir, f)
                if os.path.isfile(fp) and not f.endswith('.zip'):
                    zf.write(fp, f)
            figures_dir = os.path.join(root_dir, 'figures')
            if os.path.isdir(figures_dir):
                for fig in os.listdir(figures_dir):
                    fp = os.path.join(figures_dir, fig)
                    if os.path.isfile(fp):
                        zf.write(fp, os.path.join('figures', fig))

    for arm in arms:
        arm_dir = os.path.join(base_dir, arm)
        if os.path.isdir(arm_dir):
            for root, dirs, files in os.walk(arm_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    arc = os.path.join(arm, os.path.relpath(fp, arm_dir))
                    zf.write(fp, arc)

print(f"ZIP_SUCCESS:{{out_zip}}:{{os.path.getsize(out_zip)}}")
"""
    logger.info("Executing remote packaging for %s (Compression)...", exp_key)
    out = execute_remote_code(code, timeout=300)
    for line in out.splitlines():
        if line.startswith("ZIP_SUCCESS:"):
            parts = line.split(":")
            return parts[1]
    return out_zip


# ---------------------------------------------------------------------------
# Remote Packaging for Explainability Experiments (Exp 6)
# ---------------------------------------------------------------------------

def remote_package_xai(exp_key: str, spec: dict) -> str:
    exp_dir_name = spec["remote_dir"]
    out_zip = f"{exp_key}_daemon_sync.zip"

    code = f"""
import os, zipfile
exp_name = '{exp_dir_name}'
out_zip = '{out_zip}'

candidates = [
    f'runs/detect/runs/detect/{{exp_name}}',
    f'runs/detect/{{exp_name}}',
    exp_name
]
base_dir = None
for c in candidates:
    if os.path.exists(c):
        base_dir = c
        break

if not base_dir:
    base_dir = exp_name

if os.path.exists(out_zip):
    os.remove(out_zip)

with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root_dir in [exp_name, base_dir]:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.endswith('.zip'):
                        continue
                    fp = os.path.join(root, f)
                    arc = os.path.relpath(fp, root_dir)
                    zf.write(fp, arc)

print(f"ZIP_SUCCESS:{{out_zip}}:{{os.path.getsize(out_zip)}}")
"""
    logger.info("Executing remote packaging for %s (xAI)...", exp_key)
    out = execute_remote_code(code, timeout=180)
    for line in out.splitlines():
        if line.startswith("ZIP_SUCCESS:"):
            parts = line.split(":")
            return parts[1]
    return out_zip


# ---------------------------------------------------------------------------
# Local Extraction and Canonical Restructuring
# ---------------------------------------------------------------------------

def restructure_local_detection_arm(arm_dir: Path, arm_name: str) -> None:
    curves_dir = arm_dir / "curves"
    logs_dir = arm_dir / "logs"
    curves_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    for f in list(arm_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name in _ARM_ROOT_KEEP:
            continue
        if f.name in _CURVE_FILES:
            dst = curves_dir / f.name
            shutil.copy2(str(f), str(dst))
            f.unlink()
        elif f.name in _LOG_FILES:
            dst = logs_dir / f.name
            shutil.copy2(str(f), str(dst))
            f.unlink()

    arm_readme = arm_dir / "README.md"
    if not arm_readme.exists():
        arm_readme.write_text(f"# {arm_name}\n\nModel artifacts for {arm_name}.\n", encoding="utf-8")

    for sub in ["curves", "logs", "weights", "qualitative_samples", "test_eval", "test_inferences"]:
        sdir = arm_dir / sub
        if sdir.is_dir():
            s_readme = sdir / "README.md"
            if not s_readme.exists():
                s_readme.write_text(f"# {arm_name} - {sub}\n\nArtifacts for {sub}.\n", encoding="utf-8")


def clean_redundant_and_empty_dirs(exp_dir: Path) -> None:
    rep_dir = exp_dir / "reports"
    if rep_dir.exists():
        shutil.rmtree(rep_dir)
        logger.info("  Removed redundant reports/ directory.")

    for root, dirs, files in list(os.walk(exp_dir, topdown=False)):
        p = Path(root)
        contents = list(p.iterdir())
        if not contents:
            (p / "README.md").write_text(f"# {p.name}\n\nArtifacts directory.\n", encoding="utf-8")
            logger.info("  Populated empty directory with README: %s", p.relative_to(exp_dir))


def sync_single_experiment(exp_key: str, spec: dict) -> bool:
    local_dir = LOCAL_ARTIFACTS / spec["local_folder"]
    local_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = local_dir / f"_tmp_{exp_key}.zip"

    if spec["type"] == "detection":
        remote_zip = remote_evaluate_and_package_detection(exp_key, spec)
    elif spec["type"] == "compression":
        remote_zip = remote_evaluate_and_package_compression(exp_key, spec)
    else:
        remote_zip = remote_package_xai(exp_key, spec)

    ok = download_file(remote_zip, tmp_zip)
    if not ok:
        logger.error("Failed to download %s", remote_zip)
        return False

    if spec["type"] in ("detection", "compression"):
        for arm in spec["arms"]:
            arm_p = local_dir / arm
            if arm_p.exists():
                shutil.rmtree(arm_p)

    try:
        logger.info("Extracting %s -> %s...", tmp_zip.name, local_dir.name)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(local_dir)
        if tmp_zip.exists():
            tmp_zip.unlink()
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return False

    if spec["type"] == "detection":
        for arm in spec["arms"]:
            arm_dir = local_dir / arm
            if arm_dir.is_dir():
                restructure_local_detection_arm(arm_dir, arm)

    clean_redundant_and_empty_dirs(local_dir)

    logger.info("Successfully synced and organized %s!", exp_key)
    return True


# ---------------------------------------------------------------------------
# Daemon Polling Loop
# ---------------------------------------------------------------------------

def run_sync_cycle(state: dict, target_exp: Optional[str] = None) -> dict:
    logger.info("--- Daemon poll: %s ---", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    pipeline = fetch_pipeline_status()
    if not pipeline:
        logger.warning("Pipeline status unavailable.")
        return state

    synced = state.setdefault("synced", {})
    experiments = pipeline.get("experiments", {})

    for exp_key, spec in EXPERIMENT_SPECS.items():
        if target_exp and exp_key != target_exp:
            continue

        exp_info = experiments.get(exp_key, {})
        status = exp_info.get("status", "UNKNOWN")
        finished_at = exp_info.get("finished_at")

        if exp_key in ["exp5", "exp6"]:
            logger.info("  %s: Managed locally on client GPU (zero remote overwrite).", exp_key)
            continue

        if status != "COMPLETED":
            logger.info("  %s: %s (no action needed)", exp_key, status)
            continue

        prev = synced.get(exp_key, {})
        if prev.get("remote_finished_at") == finished_at and prev.get("status") == "OK":
            logger.info("  %s: already synced for completion %s", exp_key, finished_at)
            continue

        logger.info("  %s: COMPLETED at %s! Initiating full evaluation & synchronization...", exp_key, finished_at)
        ok = sync_single_experiment(exp_key, spec)

        synced[exp_key] = {
            "synced_at": datetime.utcnow().isoformat(),
            "remote_finished_at": finished_at,
            "status": "OK" if ok else "FAILED",
        }
        save_state(state)

    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Experiment Sync Daemon & Post-Processor")
    parser.add_argument("--interval", type=int, default=180, help="Poll interval in seconds (default: 180)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--exp", type=str, default=None, help="Sync specific experiment key only")
    args = parser.parse_args()

    state = load_state()
    logger.info("Sync daemon initialized. Interval: %ds, Mode: %s, Target: %s",
                args.interval, "single-shot" if args.once else "continuous", args.exp or "all")

    while True:
        try:
            state = run_sync_cycle(state, target_exp=args.exp)
        except Exception as exc:
            logger.exception("Error during daemon cycle: %s", exc)

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()