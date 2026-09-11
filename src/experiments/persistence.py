"""Unified Experiment Artifact Persistence and Directory Management Engine.

Guarantees complete preservation of training telemetry, model weights,
diagnostic loss/PR curves, zero-leakage test evaluations, and test inference visual samples.
Enforces strict hierarchical folder structure without loose duplicate root files.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.config import ProjectPaths
from src.experiments.schema import ArmConfig, RunMetrics, TestInferenceSample

logger = logging.getLogger("ExperimentPersistence")


class ExperimentArtifactManager:
    """Manages physical directory layout and artifact persistence for an experiment.

    Enforces standardized directory layout:
    artifacts/experiments/<exp_id>/
    ├── figures/
    ├── reports/
    └── <arm_id>/
        ├── weights/
        ├── curves/
        ├── logs/
        ├── test_eval/
        └── test_inferences/

    Attributes:
        exp_id: Identifier of the experiment (e.g. 'exp1_architecture_battle').
        paths: Root project configuration paths container.
        exp_dir: Absolute path to the experiment storage root.
    """

    def __init__(self, exp_id: str, paths: Optional[ProjectPaths] = None) -> None:
        """Initializes manager with experiment identifier and project paths.

        Args:
            exp_id: Machine-readable experiment identifier.
            paths: Standardized project configuration paths container.
        """
        self.exp_id = exp_id
        self.paths = paths or ProjectPaths()
        self.exp_dir = self.paths.experiments_dir / exp_id
        self.figures_dir = self.exp_dir / "figures"
        self.reports_dir = self.exp_dir / "reports"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_arm_dir(self, arm_id: str) -> Path:
        """Returns the canonical directory for a specific arm.

        Args:
            arm_id: Identifier of the model arm.

        Returns:
            Path pointing to the arm root directory.
        """
        return self.exp_dir / arm_id

    def ensure_arm_directories(self, arm_id: str) -> Dict[str, Path]:
        """Creates and verifies the complete standardized subfolder structure for an arm.

        Args:
            arm_id: Identifier of the model arm.

        Returns:
            Dictionary mapping category keys ('root', 'weights', 'curves', 'logs',
            'test_eval', 'test_inferences') to their verified directory paths.
        """
        arm_dir = self.get_arm_dir(arm_id)
        subdirs = {
            "root": arm_dir,
            "weights": arm_dir / "weights",
            "curves": arm_dir / "curves",
            "logs": arm_dir / "logs",
            "test_eval": arm_dir / "test_eval",
            "test_inferences": arm_dir / "test_inferences",
        }
        for path in subdirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return subdirs

    def save_run_artifacts(
        self,
        arm_id: str,
        metrics: RunMetrics,
        source_dir: Optional[Path] = None,
        weights_source: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Consolidates and archives all generated artifacts for a finished arm run.

        Moves and organizes:
        - Weights (best.pt, last.pt) -> <arm_id>/weights/
        - Curves (results.png, confusion matrices, PR curves) -> <arm_id>/curves/
        - Logs (results.csv, args.yaml, train.log) -> <arm_id>/logs/
        - Metrics summary -> <arm_id>/metrics_summary.json

        Args:
            arm_id: Identifier of the model arm.
            metrics: Structured dataclass housing quantitative telemetry.
            source_dir: Optional source directory where training output files reside.
            weights_source: Optional directory where weights were saved.

        Returns:
            Dictionary mapping artifact categories to their saved file paths.
        """
        dirs = self.ensure_arm_directories(arm_id)
        saved_paths: Dict[str, Path] = {}

        # 1. Metrics JSON serialization
        metrics_dict = {
            "arm_id": metrics.arm_id,
            "train_duration_sec": metrics.train_duration_sec,
            "epochs_completed": metrics.epochs_completed,
            "val_map50": metrics.val_map50,
            "val_map50_95": metrics.val_map50_95,
            "val_precision": metrics.val_precision,
            "val_recall": metrics.val_recall,
            "test_map50": metrics.test_map50,
            "test_map50_95": metrics.test_map50_95,
            "test_precision": metrics.test_precision,
            "test_recall": metrics.test_recall,
            "parameters_m": metrics.parameters_m,
            "gflops": metrics.gflops,
            "latency_ms": metrics.latency_ms,
            "fps": metrics.fps,
            "model_size_mb": metrics.model_size_mb,
            "per_class_test_ap": metrics.per_class_test_ap,
        }
        summary_path = dirs["root"] / "metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        saved_paths["metrics_summary"] = summary_path

        # 2. File migration from training output directory if provided
        if source_dir and source_dir.exists():
            # Migrate logs
            for log_file in ["results.csv", "args.yaml", "train.log"]:
                src = source_dir / log_file
                if src.exists():
                    dst = dirs["logs"] / log_file
                    shutil.copy2(src, dst)
                    saved_paths[log_file] = dst

            # Migrate curves & figures
            curve_patterns = [
                "results.png",
                "confusion_matrix*.png",
                "Box*.png",
                "labels*.jpg",
                "val_batch*.jpg",
                "train_batch*.jpg",
            ]
            for pattern in curve_patterns:
                for src in source_dir.glob(pattern):
                    dst = dirs["curves"] / src.name
                    shutil.copy2(src, dst)

            # Migrate qualitative predictions / test inferences
            qual_dir = source_dir / "qualitative_samples"
            if qual_dir.exists():
                for qf in qual_dir.glob("*.*"):
                    dst = dirs["test_inferences"] / qf.name
                    shutil.copy2(qf, dst)

        # 3. Migrate weights
        w_dir = weights_source or (source_dir / "weights" if source_dir else None)
        if w_dir and w_dir.exists():
            for w_name in ["best.pt", "last.pt", "best.onnx", "best.tflite"]:
                src_w = w_dir / w_name
                if src_w.exists():
                    dst_w = dirs["weights"] / w_name
                    shutil.copy2(src_w, dst_w)
                    saved_paths[w_name] = dst_w

        logger.info("Successfully persisted all training artifacts for arm '%s' into %s", arm_id, dirs["root"])
        return saved_paths

    def save_test_inferences(
        self,
        arm_id: str,
        samples: List[TestInferenceSample],
    ) -> Path:
        """Persists quantitative test inference detection records and JSON manifest.

        Args:
            arm_id: Identifier of the model arm.
            samples: List of TestInferenceSample objects.

        Returns:
            Path pointing to the written test inferences JSON manifest.
        """
        dirs = self.ensure_arm_directories(arm_id)
        manifest_payload = []
        for s in samples:
            manifest_payload.append({
                "image_filename": s.image_filename,
                "predictions": s.predictions,
                "ground_truths": s.ground_truths,
                "is_exact_match": s.is_exact_match,
            })

        out_path = dirs["test_inferences"] / "inference_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)
        logger.info("Saved %d test inference sample records to %s", len(samples), out_path)
        return out_path

    def save_summary_tables(
        self,
        summary_df: pd.DataFrame,
        summary_records: List[Dict[str, Any]],
        basename: str,
    ) -> Tuple[Path, Path]:
        """Saves consolidated tabular metrics across all arms to CSV and JSON.

        Args:
            summary_df: Pandas DataFrame containing consolidated arm comparisons.
            summary_records: List of dictionaries matching the DataFrame rows.
            basename: Base filename without extension (e.g. 'exp1_architecture_comparison').

        Returns:
            Tuple of (csv_path, json_path).
        """
        csv_path = self.exp_dir / f"{basename}.csv"
        json_path = self.exp_dir / f"{basename}.json"

        summary_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=2)
        # Also mirror in reports/ subdirectory for clean documentation
        shutil.copy2(csv_path, self.reports_dir / f"{basename}.csv")
        shutil.copy2(json_path, self.reports_dir / f"{basename}.json")

        logger.info("Consolidated experiment tables saved to %s and %s", csv_path.name, json_path.name)
        return csv_path, json_path

    def export_summary(
        self,
        records: List[Dict[str, Any]],
        base_name: str,
    ) -> Tuple[Path, Path]:
        """Saves list of summary records to CSV and JSON.

        Args:
            records: List of dictionaries matching the arm comparison rows.
            base_name: Base filename without extension.

        Returns:
            Tuple of (csv_path, json_path).
        """
        df = pd.DataFrame(records)
        return self.save_summary_tables(df, records, base_name)

    def clean_duplicate_root_figures(self) -> List[Path]:
        """Detects and resolves loose figure files sitting directly in the experiment root.

        Files ending in .png sitting directly under artifacts/experiments/<exp_id>/
        are moved to <exp_id>/figures/ if not already present, or removed if duplicate.

        Returns:
            List of processed file paths.
        """
        processed: List[Path] = []
        for f in self.exp_dir.glob("*.png"):
            fig_sub_path = self.figures_dir / f.name
            if not fig_sub_path.exists():
                shutil.move(str(f), str(fig_sub_path))
                processed.append(fig_sub_path)
                logger.info("Moved loose root figure %s into %s", f.name, self.figures_dir)
            else:
                f.unlink(missing_ok=True)
                processed.append(fig_sub_path)
                logger.info("Pruned duplicate root figure %s", f.name)
        return processed

    def verify_structure(self) -> Dict[str, Any]:
        """Audits the experiment directory and verifies standardized layout.

        Returns:
            Structured dictionary summarizing arm subfolder completeness and figure counts.
        """
        subfolders = [p for p in self.exp_dir.iterdir() if p.is_dir() and p.name not in ["figures", "reports"]]
        arm_status = {}
        for arm in subfolders:
            arm_status[arm.name] = {
                "has_weights": (arm / "weights").exists() and any((arm / "weights").iterdir()),
                "has_curves": (arm / "curves").exists() and any((arm / "curves").iterdir()),
                "has_logs": (arm / "logs").exists() and any((arm / "logs").iterdir()),
                "has_test_eval": (arm / "test_eval").exists(),
                "has_test_inferences": (arm / "test_inferences").exists(),
            }

        figures_count = len(list(self.figures_dir.glob("*.png")))
        loose_pngs = len(list(self.exp_dir.glob("*.png")))

        return {
            "exp_id": self.exp_id,
            "arms_discovered": len(subfolders),
            "arm_status": arm_status,
            "figures_in_figures_dir": figures_count,
            "loose_root_figures": loose_pngs,
            "is_clean": loose_pngs == 0,
        }
