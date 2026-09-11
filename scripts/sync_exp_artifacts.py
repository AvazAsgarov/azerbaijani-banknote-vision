"""Artifact Synchronizer for Azerbaijani Banknote Experiments 1-6.

Downloads and extracts complete experiment bundles from the remote A100 server
into standardized local directory hierarchies:
- artifacts/experiments/expX_name/
  ├── figures/
  ├── reports/
  └── <arm_id>/
      ├── weights/
      ├── curves/
      ├── logs/
      ├── test_eval/
      └── test_inferences/
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths, RemoteClusterConfig
from src.experiments.persistence import ExperimentArtifactManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ArtifactSync")

CLUSTER_CFG = RemoteClusterConfig()
BASE_URL = CLUSTER_CFG.base_url.rstrip("/")
TOKEN = CLUSTER_CFG.token
HEADERS = CLUSTER_CFG.get_auth_headers()

EXP_MAPPINGS = {
    "exp1": {
        "remote_complete_zip": "exp1_architecture_battle_complete.zip",
        "remote_legacy_zip": "exp1_results_bundle.zip",
        "remote_csv": "exp1_architecture_battle/exp1_architecture_comparison.csv",
        "remote_json": "exp1_architecture_battle/exp1_architecture_comparison.json",
        "remote_fig": "exp1_architecture_battle/figures/pareto_latency_vs_map.png",
        "local_name": "exp1_architecture_battle",
    },
    "exp2": {
        "remote_complete_zip": "exp2_data_augmentation_complete.zip",
        "remote_legacy_zip": "exp2_results_bundle.zip",
        "remote_csv": "exp2_data_augmentation/exp2_augmentation_comparison.csv",
        "remote_json": "exp2_data_augmentation/exp2_augmentation_comparison.json",
        "remote_fig": "exp2_data_augmentation/figures/augmentation_ablation_delta.png",
        "local_name": "exp2_data_augmentation",
    },
    "exp3": {
        "remote_complete_zip": "exp3_color_space_shortcut_complete.zip",
        "remote_legacy_zip": "exp3_results_bundle.zip",
        "remote_csv": "exp3_color_space_shortcut/exp3_color_space_comparison.csv",
        "remote_json": "exp3_color_space_shortcut/exp3_color_space_comparison.json",
        "remote_fig": "exp3_color_space_shortcut/figures/color_space_shortcut_delta.png",
        "local_name": "exp3_color_space_shortcut",
    },
    "exp4": {
        "remote_complete_zip": "exp4_resolution_scaling_complete.zip",
        "remote_legacy_zip": "exp4_results_bundle.zip",
        "remote_csv": "exp4_multiscale_resolution/exp4_resolution_comparison.csv",
        "remote_json": "exp4_multiscale_resolution/exp4_resolution_comparison.json",
        "remote_fig": "exp4_multiscale_resolution/figures/resolution_vs_map_latency.png",
        "local_name": "exp4_resolution_scaling",
    },
    "exp5": {
        "remote_complete_zip": "exp5_model_compression_complete.zip",
        "remote_legacy_zip": "exp5_results_bundle.zip",
        "remote_csv": "exp5_compression_optimization/exp5_compression_comparison.csv",
        "remote_json": "exp5_compression_optimization/exp5_compression_comparison.json",
        "remote_fig": "exp5_compression_optimization/figures/compression_pareto_frontier.png",
        "local_name": "exp5_model_compression",
    },
    "exp6": {
        "remote_complete_zip": "exp6_explainability_xai_complete.zip",
        "remote_legacy_zip": "exp6_results_bundle.zip",
        "remote_csv": "exp6_explainable_ai/exp6_xai_summary.csv",
        "remote_json": "exp6_explainable_ai/exp6_xai_summary.json",
        "remote_fig": "exp6_explainable_ai/numismatic_alignment_by_denomination.png",
        "local_name": "exp6_explainability_xai",
    },
}


def sync_experiment(exp_key: str, paths: Optional[ProjectPaths] = None) -> bool:
    """Synchronizes full experiment artifacts from remote server into local storage.

    Args:
        exp_key: Experiment key ('exp1' to 'exp6').
        paths: Standardized ProjectPaths instance.

    Returns:
        True if sync succeeded, False otherwise.
    """
    proj_paths = paths or ProjectPaths()
    cfg = EXP_MAPPINGS[exp_key]
    exp_name = cfg["local_name"]
    local_dir = proj_paths.experiments_dir / exp_name
    local_dir.mkdir(parents=True, exist_ok=True)

    manager = ExperimentArtifactManager(exp_name, paths=proj_paths)

    # Prefer comprehensive bundle; fallback to legacy bundle
    zip_candidates = [cfg["remote_complete_zip"], cfg["remote_legacy_zip"]]
    downloaded_zip = None

    for z_name in zip_candidates:
        try:
            r_check = requests.get(f"{BASE_URL}/api/contents/{z_name}?content=0", headers=HEADERS, timeout=8)
            if r_check.status_code == 200:
                downloaded_zip = z_name
                break
        except Exception as e:
            logger.debug("Check failed for %s: %s", z_name, e)

    if downloaded_zip:
        logger.info("Downloading %s for %s via streaming...", downloaded_zip, exp_key)
        download_url = f"{BASE_URL}/files/{downloaded_zip}?token={TOKEN}"
        try:
            with requests.get(download_url, headers=HEADERS, stream=True, timeout=180) as r_stream:
                r_stream.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    for chunk in r_stream.iter_content(chunk_size=2 * 1024 * 1024):
                        if chunk:
                            tmp_file.write(chunk)

            logger.info("Extracting %s (%.1f MB) into %s...", downloaded_zip, tmp_path.stat().st_size / (1024 * 1024), local_dir)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(local_dir)
            tmp_path.unlink(missing_ok=True)
            logger.info("Archive extracted successfully.")
        except Exception as exc:
            logger.error("Failed to stream download or extract %s: %s", downloaded_zip, exc)
    else:
        logger.warning("No zip archive found for %s on remote server", exp_key)

    # Synchronize summary files (CSV/JSON into reports/ and root)
    for item_key in ["remote_csv", "remote_json"]:
        remote_item = cfg.get(item_key)
        if remote_item:
            try:
                r = requests.get(f"{BASE_URL}/files/{remote_item}?token={TOKEN}", headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    item_name = Path(remote_item).name
                    (local_dir / item_name).write_bytes(r.content)
                    (manager.reports_dir / item_name).write_bytes(r.content)
                    logger.info("Synced summary file: %s", item_name)
            except Exception as e:
                logger.warning("Could not sync %s: %s", remote_item, e)

    # Synchronize figure directly into figures/
    remote_fig = cfg.get("remote_fig")
    if remote_fig:
        try:
            r = requests.get(f"{BASE_URL}/files/{remote_fig}?token={TOKEN}", headers=HEADERS, timeout=15)
            if r.status_code == 200:
                fig_dest = manager.figures_dir / Path(remote_fig).name
                fig_dest.write_bytes(r.content)
                logger.info("Synced figure into figures/: %s", fig_dest.name)
        except Exception as e:
            logger.warning("Could not sync figure %s: %s", remote_fig, e)

    # Ensure all arms have complete subfolder structure
    for child in local_dir.iterdir():
        if child.is_dir() and child.name not in ["figures", "reports", "heatmaps"]:
            manager.ensure_arm_directories(child.name)

    # Clean loose duplicate figures in root
    manager.clean_duplicate_root_figures()

    audit = manager.verify_structure()
    logger.info("Sync verified for %s: %d arms, clean=%s", exp_key, audit["arms_discovered"], audit["is_clean"])
    return True


def main() -> None:
    """CLI orchestrator synchronizing experiment artifacts."""
    import argparse

    parser = argparse.ArgumentParser(description="Synchronize experiment artifacts from remote GPU cluster")
    parser.add_argument(
        "--exp",
        type=str,
        choices=list(EXP_MAPPINGS.keys()) + ["all"],
        default="all",
        help="Experiment to sync (exp1..exp6, or 'all')",
    )
    args = parser.parse_args()

    paths = ProjectPaths()
    if args.exp == "all":
        for k in sorted(EXP_MAPPINGS.keys()):
            logger.info("=== Syncing %s ===", k)
            sync_experiment(k, paths=paths)
    else:
        logger.info("=== Syncing %s ===", args.exp)
        sync_experiment(args.exp, paths=paths)

    logger.info("Experiment synchronization completed.")


if __name__ == "__main__":
    main()
