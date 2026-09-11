"""Deployment and Orchestration Script for Master Autonomous Experiment Pipeline.

Deploys `run_master_pipeline.py` to the dedicated remote A100 GPU server and launches it
as an independent, detached background daemon. Sequentially runs Experiments 1 through 6
uninterrupted, completely decoupled from client workstations and browser sessions.
"""

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import websocket

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import RemoteClusterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DeployMasterPipeline")

# The master autonomous orchestrator code to run on the remote Linux host
MASTER_PIPELINE_CODE = r'''"""
Autonomous Master Experiment Pipeline (Experiments 1 through 6).
Executes sequentially on the dedicated NVIDIA A100 GPU cluster.
Runs as an independent OS background daemon (PID parent = 1 / init).
Zero reliance on client workstation, VPN, or browser sessions.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import torch

BASE_DIR = Path("/sdb-disk/notebooks/team2")
os.chdir(BASE_DIR)

STATUS_FILE = BASE_DIR / "master_pipeline_status.json"
LOG_FILE = BASE_DIR / "master_pipeline.log"

EXPERIMENTS = [
    {
        "id": "exp1",
        "index": 1,
        "title": "Experiment 1: Architecture Battle (YOLOv8m, YOLO11m, RT-DETR-L)",
        "script": "run_experiment_1.py",
        "log_file": "run_experiment_1.log",
        "bundle": "exp1_results_bundle.zip",
        "existing_pid": 410309,
    },
    {
        "id": "exp2",
        "index": 2,
        "title": "Experiment 2: Data Augmentation Ablation (Raw, Geometric, Photometric, Full)",
        "script": "run_experiment_2.py",
        "log_file": "run_experiment_2.log",
        "bundle": "exp2_results_bundle.zip",
        "existing_pid": None,
    },
    {
        "id": "exp3",
        "index": 3,
        "title": "Experiment 3: Color Space & Shortcut Learning (RGB, Grayscale, HSV)",
        "script": "run_experiment_3.py",
        "log_file": "run_experiment_3.log",
        "bundle": "exp3_results_bundle.zip",
        "existing_pid": None,
    },
    {
        "id": "exp4",
        "index": 4,
        "title": "Experiment 4: Multiscale Resolution Dynamics (320, 640, 1280)",
        "script": "run_experiment_4.py",
        "log_file": "run_experiment_4.log",
        "bundle": "exp4_results_bundle.zip",
        "existing_pid": None,
    },
    {
        "id": "exp5",
        "index": 5,
        "title": "Experiment 5: Model Compression & Edge Optimization (FP32, FP16, INT8, Pruned)",
        "script": "run_experiment_5.py",
        "log_file": "run_experiment_5.log",
        "bundle": "exp5_results_bundle.zip",
        "existing_pid": None,
    },
    {
        "id": "exp6",
        "index": 6,
        "title": "Experiment 6: Explainable AI & Attention Diagnostics (C2PSA EigenCAM)",
        "script": "run_experiment_6.py",
        "log_file": "run_experiment_6.log",
        "bundle": "exp6_results_bundle.zip",
        "existing_pid": None,
    },
]


def log(msg: str) -> None:
    """Logs message with UTC timestamp and immediate stdout flush."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [MasterPipeline] {msg}", flush=True)


def update_status_file(status_dict: dict) -> None:
    """Persists live orchestrator state to disk for external monitoring."""
    status_dict["last_updated"] = datetime.utcnow().isoformat()
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as exc:
        log(f"Warning: Failed to update status file: {exc}")


def is_pid_alive(pid: int) -> bool:
    """Checks whether a process PID is currently active and non-zombie."""
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def clean_gpu_memory() -> None:
    """Triggers PyTorch garbage collection and CUDA cache clearing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    """Master sequential execution loop for Experiments 1 through 6."""
    log("================================================================================")
    log("MASTER AUTONOMOUS PIPELINE ORCHESTRATOR LAUNCHED")
    log(f"Operating Directory: {BASE_DIR}")
    log(f"Process PID: {os.getpid()} (Parent: {os.getppid()})")
    log("Managing Experiments 1 through 6 sequentially.")
    log("================================================================================")

    pipeline_state = {
        "pipeline_active": True,
        "master_pid": os.getpid(),
        "started_at": datetime.utcnow().isoformat(),
        "current_stage_index": 1,
        "current_stage_id": "exp1",
        "overall_status": "RUNNING",
        "experiments": {},
    }

    for exp in EXPERIMENTS:
        pipeline_state["experiments"][exp["id"]] = {
            "index": exp["index"],
            "title": exp["title"],
            "script": exp["script"],
            "log_file": exp["log_file"],
            "status": "QUEUED",
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "bundle": exp["bundle"],
        }

    update_status_file(pipeline_state)

    # -------------------------------------------------------------------------
    # STAGE 1: Monitor Experiment 1 (Already Active in Background)
    # -------------------------------------------------------------------------
    exp1_cfg = EXPERIMENTS[0]
    exp1_pid = exp1_cfg["existing_pid"]
    log(f"STAGE 1: Attaching to active Experiment 1 process (PID {exp1_pid})...")

    pipeline_state["current_stage_index"] = 1
    pipeline_state["current_stage_id"] = "exp1"
    pipeline_state["experiments"]["exp1"]["status"] = "RUNNING"
    pipeline_state["experiments"]["exp1"]["pid"] = exp1_pid
    pipeline_state["experiments"]["exp1"]["started_at"] = datetime.utcnow().isoformat()
    update_status_file(pipeline_state)

    check_count = 0
    while is_pid_alive(exp1_pid):
        check_count += 1
        if check_count % 8 == 0:  # Log every 2 minutes
            log(f"Experiment 1 is actively training on A100 GPU (PID {exp1_pid}).")
        time.sleep(15)

    log(f"Experiment 1 process (PID {exp1_pid}) has terminated.")
    pipeline_state["experiments"]["exp1"]["status"] = "COMPLETED"
    pipeline_state["experiments"]["exp1"]["finished_at"] = datetime.utcnow().isoformat()
    update_status_file(pipeline_state)

    clean_gpu_memory()
    log("Experiment 1 complete. GPU memory cleared. Pausing 15s before next stage...")
    time.sleep(15)

    # -------------------------------------------------------------------------
    # STAGES 2 THROUGH 6: Sequential Execution
    # -------------------------------------------------------------------------
    for exp_cfg in EXPERIMENTS[1:]:
        exp_id = exp_cfg["id"]
        exp_idx = exp_cfg["index"]
        exp_title = exp_cfg["title"]
        script_name = exp_cfg["script"]
        log_name = exp_cfg["log_file"]
        bundle_name = exp_cfg["bundle"]

        log("================================================================================")
        log(f"STARTING STAGE {exp_idx} / 6: {exp_title}")
        log(f"Script: {script_name} | Logging to: {log_name}")
        log("================================================================================")

        pipeline_state["current_stage_index"] = exp_idx
        pipeline_state["current_stage_id"] = exp_id
        pipeline_state["experiments"][exp_id]["status"] = "RUNNING"
        pipeline_state["experiments"][exp_id]["started_at"] = datetime.utcnow().isoformat()
        update_status_file(pipeline_state)

        # Clear previous zip bundle to ensure fresh verification
        bundle_path = BASE_DIR / bundle_name
        if bundle_path.exists():
            try:
                bundle_path.unlink()
                log(f"Removed prior bundle {bundle_name} to guarantee fresh output.")
            except Exception as e:
                log(f"Notice: Could not remove old bundle: {e}")

        # Launch script with unbuffered output redirected to log file
        log_path = BASE_DIR / log_name
        with open(log_path, "w", encoding="utf-8", buffering=1) as log_fp:
            proc = subprocess.Popen(
                [sys.executable, "-u", script_name],
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
            )
            pipeline_state["experiments"][exp_id]["pid"] = proc.pid
            update_status_file(pipeline_state)
            log(f"Spawned {script_name} with PID {proc.pid}. Awaiting execution completion...")

            return_code = proc.wait()

        clean_gpu_memory()

        if return_code == 0:
            log(f"Stage {exp_idx} ({exp_title}) COMPLETED SUCCESSFULLY (code 0).")
            pipeline_state["experiments"][exp_id]["status"] = "COMPLETED"
        else:
            log(f"Stage {exp_idx} ({exp_title}) EXITED WITH WARNING/CODE {return_code}.")
            pipeline_state["experiments"][exp_id]["status"] = f"COMPLETED_WITH_WARNINGS (code {return_code})"

        pipeline_state["experiments"][exp_id]["finished_at"] = datetime.utcnow().isoformat()
        update_status_file(pipeline_state)

        log("Post-stage cooldown: 15 seconds before launching next experiment...")
        time.sleep(15)

    # -------------------------------------------------------------------------
    # PIPELINE FINALIZATION
    # -------------------------------------------------------------------------
    log("================================================================================")
    log("ALL 6 EXPERIMENTS COMPLETED IN AUTONOMOUS SEQUENCE!")
    log("================================================================================")

    # Bundle all experiment folders into master bundle
    master_zip_name = "all_experiments_master_bundle"
    try:
        log("Creating all-inclusive archive of all experiment results...")
        target_dirs = [
            "exp1_architecture_battle",
            "exp2_data_augmentation",
            "exp3_color_space_shortcut",
            "exp4_multiscale_resolution",
            "exp5_compression_optimization",
            "exp6_explainable_ai",
        ]
        combined_dir = BASE_DIR / "all_experiments_summary"
        combined_dir.mkdir(exist_ok=True)
        for td in target_dirs:
            src_p = BASE_DIR / td
            dst_p = combined_dir / td
            if src_p.exists() and not dst_p.exists():
                shutil.copytree(src_p, dst_p, ignore=shutil.ignore_patterns("*.pt"))

        shutil.make_archive(str(BASE_DIR / master_zip_name), "zip", str(combined_dir))
        log(f"Master archive created: {master_zip_name}.zip")
    except Exception as exc:
        log(f"Notice during master bundle creation: {exc}")

    pipeline_state["pipeline_active"] = False
    pipeline_state["overall_status"] = "COMPLETED"
    update_status_file(pipeline_state)
    log("Master autonomous pipeline finished successfully. Daemon exiting cleanly.")


if __name__ == "__main__":
    main()
'''


def deploy_and_launch() -> bool:
    """Deploys the orchestrator script to remote server and launches detached daemon.

    Returns:
        True if successfully deployed and launched, False otherwise.
    """
    cfg = RemoteClusterConfig()
    base_url = cfg.base_url.rstrip("/")
    headers = cfg.get_auth_headers()

    logger.info("Uploading run_master_pipeline.py to remote GPU server via Contents API...")
    save_url = f"{base_url}/api/contents/run_master_pipeline.py"
    payload = {
        "type": "file",
        "format": "text",
        "content": MASTER_PIPELINE_CODE,
    }
    resp = requests.put(save_url, headers=headers, json=payload, timeout=30)
    if resp.status_code not in (200, 201):
        logger.error("Failed to upload run_master_pipeline.py: %s", resp.text)
        return False
    logger.info("run_master_pipeline.py uploaded successfully.")

    # Launch daemon detached using a temporary Jupyter kernel
    logger.info("Spawning temporary kernel to launch master daemon with start_new_session=True...")
    kernel_resp = requests.post(
        f"{base_url}/api/kernels",
        headers=headers,
        json={"name": "python3"},
        timeout=30,
    )
    if kernel_resp.status_code not in (200, 201):
        logger.error("Failed to create kernel for launching daemon: %s", kernel_resp.text)
        return False

    kernel_id = kernel_resp.json()["id"]
    ws_url = cfg.get_websocket_url(kernel_id)
    ws = websocket.create_connection(ws_url, timeout=30)

    launcher_code = """
import subprocess
import sys

cmd = [sys.executable, '-u', 'run_master_pipeline.py']
log_fp = open('master_pipeline.log', 'w', buffering=1)
proc = subprocess.Popen(
    cmd,
    stdout=log_fp,
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print(f"MASTER_DAEMON_PID={proc.pid}")
"""

    msg_id = uuid.uuid4().hex
    exec_payload = {
        "header": {
            "msg_id": msg_id,
            "username": "client",
            "session": msg_id,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "metadata": {},
        "content": {
            "code": launcher_code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
        },
        "parent_header": {},
        "channel": "shell",
    }
    ws.send(json.dumps(exec_payload))

    daemon_pid = None
    while True:
        raw = ws.recv()
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") == msg_id:
            mtype = msg.get("msg_type")
            if mtype == "stream":
                txt = msg["content"]["text"]
                logger.info("[Remote Kernel] %s", txt.strip())
                if "MASTER_DAEMON_PID=" in txt:
                    daemon_pid = txt.split("MASTER_DAEMON_PID=")[1].strip()
            elif mtype == "status" and msg.get("content", {}).get("execution_state") == "idle":
                break

    ws.close()
    requests.delete(f"{base_url}/api/kernels/{kernel_id}", headers=headers)

    if daemon_pid:
        logger.info("Master Autonomous Pipeline successfully launched! Daemon PID: %s", daemon_pid)
        return True
    else:
        logger.error("Could not capture Master Daemon PID from remote kernel.")
        return False


if __name__ == "__main__":
    success = deploy_and_launch()
    sys.exit(0 if success else 1)
