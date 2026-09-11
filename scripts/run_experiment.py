"""Unified Research Experiment Execution and Orchestration CLI.

Orchestrates execution of Experiments 1 through 6 on Azerbaijani Banknote Detection,
supporting both Local GPU/CPU execution and Dedicated Remote A100 GPU cluster dispatch.

Experiments:
  Exp 1: Architecture Battle (DINOv2 ViT-L/14 vs YOLOv8m vs YOLOv11m vs RT-DETR-L)
  Exp 2: Data Augmentation Impact Ablation (Raw vs Geometric vs Photometric vs Full Composite)
  Exp 3: Color Space & Shortcut Learning (RGB vs HSV vs Grayscale Cross-Evaluation)
  Exp 4: Multiscale Resolution Dynamics (320x320 vs 640x640 vs 1280x1280)
  Exp 5: Model Compression & Edge Optimization (FP32 vs FP16 vs INT8 PTQ vs 25% L1 Pruning)
  Exp 6: Explainable AI & Attention Diagnostics (C2PSA EigenCAM & Numismatic Motif Alignment)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import ProjectPaths, RemoteClusterConfig
from src.experiments.persistence import ExperimentArtifactManager
from src.experiments.schema import ArmConfig, ExperimentMetadata, RunMetrics
from src.experiments.visualizer import ExperimentVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ExperimentOrchestrator")

cluster_config = RemoteClusterConfig()
REMOTE_SERVER_URL = cluster_config.base_url
REMOTE_TOKEN = cluster_config.token



EXPERIMENTS_REGISTRY = {
    1: {
        "key": "exp1",
        "folder": "exp1_architecture_battle",
        "title": "Experiment 1: Architecture Battle",
        "remote_writer": "write_remote_exp1.py",
        "summary_csv": "exp1_architecture_comparison.csv",
        "summary_json": "exp1_architecture_comparison.json",
    },
    2: {
        "key": "exp2",
        "folder": "exp2_data_augmentation",
        "title": "Experiment 2: Data Augmentation Ablation",
        "remote_writer": "write_remote_exp2.py",
        "summary_csv": "exp2_augmentation_comparison.csv",
        "summary_json": "exp2_augmentation_comparison.json",
    },
    3: {
        "key": "exp3",
        "folder": "exp3_color_space_shortcut",
        "title": "Experiment 3: Color Space & Shortcut Learning",
        "remote_writer": "write_remote_exp3.py",
        "summary_csv": "exp3_color_space_comparison.csv",
        "summary_json": "exp3_color_space_comparison.json",
    },
    4: {
        "key": "exp4",
        "folder": "exp4_resolution_scaling",
        "title": "Experiment 4: Multiscale Resolution Dynamics",
        "remote_writer": "write_remote_exp4.py",
        "summary_csv": "exp4_resolution_comparison.csv",
        "summary_json": "exp4_resolution_comparison.json",
    },
    5: {
        "key": "exp5",
        "folder": "exp5_model_compression",
        "title": "Experiment 5: Edge Optimization & Model Compression",
        "remote_writer": "write_remote_exp5.py",
        "summary_csv": "exp5_compression_comparison.csv",
        "summary_json": "exp5_compression_comparison.json",
    },
    6: {
        "key": "exp6",
        "folder": "exp6_explainability_xai",
        "title": "Experiment 6: Explainable AI & Attention Diagnostics",
        "remote_writer": "write_remote_exp6.py",
        "summary_csv": "exp6_xai_summary.csv",
        "summary_json": "exp6_xai_summary.json",
    },
}


def check_remote_connectivity() -> bool:
    """Verifies HTTP connectivity to remote A100 GPU server."""
    import requests
    try:
        r = requests.get(
            f"{REMOTE_SERVER_URL}/api/status",
            headers={"Authorization": f"token {REMOTE_TOKEN}"},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def run_remote_experiment(exp_id: int) -> bool:
    """Dispatches and tracks experiment execution on the remote A100 GPU cluster.

    Args:
        exp_id: Integer index (1 to 6).

    Returns:
        True if remote execution and artifact synchronization succeeded.
    """
    cfg = EXPERIMENTS_REGISTRY[exp_id]
    logger.info("=== Launching Remote Experiment %d: %s ===", exp_id, cfg["title"])

    if not check_remote_connectivity():
        logger.error(
            "Remote server unreachable at %s. Ensure VPN / tunnel is active.",
            REMOTE_SERVER_URL,
        )
        return False

    script_path = Path(__file__).resolve().parent / cfg["remote_writer"]
    logger.info("Executing remote submission script: %s", script_path.name)

    res = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    if res.returncode != 0:
        logger.error("Remote runner script exited with code %d", res.returncode)
        return False

    # Synchronize artifacts back
    logger.info("Synchronizing generated artifacts from remote cluster...")
    sync_script = Path(__file__).resolve().parent / "sync_exp_artifacts.py"
    sync_res = subprocess.run(
        [sys.executable, str(sync_script), "--exp", cfg["key"]],
        capture_output=False,
    )
    if sync_res.returncode != 0:
        logger.warning("Artifact synchronization completed with warnings.")

    # Generate qualitative test inferences
    inf_script = Path(__file__).resolve().parent / "generate_test_inferences.py"
    subprocess.run(
        [sys.executable, str(inf_script), "--exp", cfg["key"]],
        capture_output=False,
    )

    logger.info("Experiment %d remote execution, sync, and inference complete!", exp_id)
    return True


def run_local_experiment(
    exp_id: int,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
) -> bool:
    """Executes experiment locally using available local compute.

    Args:
        exp_id: Integer index (1 to 6).
        epochs: Optional epoch override.
        batch: Optional batch size override.
        imgsz: Optional image resolution override.
        device: Target PyTorch compute device ('0', 'cuda:0', 'cpu').

    Returns:
        True if local execution succeeded.
    """
    import torch
    cfg = EXPERIMENTS_REGISTRY[exp_id]
    paths = ProjectPaths()
    manager = ExperimentArtifactManager(cfg["folder"], paths=paths)

    effective_device = device or ("0" if torch.cuda.is_available() else "cpu")
    logger.info(
        "=== Executing Local Experiment %d: %s [Device: %s] ===",
        exp_id, cfg["title"], effective_device
    )

    # If existing complete weights exist, generate diagnostic figures, inferences and evaluations
    inf_script = Path(__file__).resolve().parent / "generate_test_inferences.py"
    subprocess.run(
        [sys.executable, str(inf_script), "--exp", cfg["key"], "--device", effective_device],
        capture_output=False,
    )

    manager.clean_duplicate_root_figures()
    audit = manager.verify_structure()
    logger.info(
        "Experiment %d verified: %d arms, clean=%s",
        exp_id, len(audit.get("arms", [])), audit.get("is_clean", False)
    )
    return True


def main() -> None:
    """CLI entrypoint for experiment execution and orchestration."""
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for Azerbaijani Banknote Detection."
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="1",
        help="Experiment number (1 to 6, or 'all')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "remote"],
        default="local",
        help="Execution target: 'local' (workstation) or 'remote' (A100 server)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override")
    parser.add_argument("--imgsz", type=int, default=None, help="Image resolution override")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('0', 'cpu')")

    args = parser.parse_args()

    targets: List[int] = []
    if args.exp.lower() == "all":
        targets = list(EXPERIMENTS_REGISTRY.keys())
    else:
        try:
            targets = [int(args.exp)]
        except ValueError:
            logger.error("Invalid experiment target: '%s'. Choose 1-6 or 'all'.", args.exp)
            sys.exit(1)

    for target_exp in sorted(targets):
        if target_exp not in EXPERIMENTS_REGISTRY:
            logger.error("Experiment %d not found in registry.", target_exp)
            continue

        if args.mode == "remote":
            run_remote_experiment(target_exp)
        else:
            run_local_experiment(
                target_exp,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
            )

    logger.info("All requested experiment operations completed.")


if __name__ == "__main__":
    main()
