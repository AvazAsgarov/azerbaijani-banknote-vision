"""Enterprise Pre-Flight Validation Gatekeeper and Pipeline Invalidation Runner.

Enforces strict data integrity, zero data leakage, and automated test coverage
prior to allocating GPU/TPU compute for deep learning model training. Any stale
artifacts, failing tests, or coverage drops below 95% abort execution immediately
with non-zero exit codes.
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root directory is registered into sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.core.config import ProjectPaths
from src.core.provenance import InvalidationStatus, ProvenanceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PreFlightGatekeeper")

COVERAGE_MINIMUM_THRESHOLD: float = 95.0


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Formats tabular data into a clean ASCII table.

    Args:
        headers: Column header strings.
        rows: Rows containing table elements.

    Returns:
        Formatted ASCII string table.
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        row_line = "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |"
        lines.append(row_line)
    lines.append(sep)
    return "\n".join(lines)


class PreFlightGatekeeper:
    """Coordinates provenance validation, automated test runs, and dataset audits.

    Attributes:
        paths: Central project paths container.
        tracker: Cryptographic provenance and invalidation tracker.
        coverage_threshold: Strict minimum line/branch coverage threshold.
    """

    def __init__(
        self,
        paths: Optional[ProjectPaths] = None,
        coverage_threshold: float = COVERAGE_MINIMUM_THRESHOLD,
    ) -> None:
        """Initializes the pre-flight gatekeeper with project configuration.

        Args:
            paths: Central paths container.
            coverage_threshold: Minimum coverage target percentage.
        """
        self.paths = paths or ProjectPaths()
        self.tracker = ProvenanceTracker(self.paths)
        self.coverage_threshold = coverage_threshold
        self.report_path = self.paths.reports_dir / "preflight_verification_report.json"

    def run_provenance_check(self) -> InvalidationStatus:
        """Evaluates artifact staleness and automatically synchronizes when mutated.

        Returns:
            InvalidationStatus describing post-synchronization state.
        """
        logger.info("=" * 70)
        logger.info("[PHASE 1/4] AUDITING ARTIFACT PROVENANCE AND DEPENDENCY DAG...")
        logger.info("=" * 70)

        status = self.tracker.evaluate_invalidation()
        if status.is_fresh:
            logger.info("Provenance Check: ALL ARTIFACTS CONFIRMED FRESH.")
            return status

        logger.warning(
            "Staleness Detected across %d stage(s): %s",
            len(status.stale_stages),
            ", ".join(status.stale_stages),
        )
        for reason in status.reasons:
            logger.warning("  -> %s", reason)

        logger.info("Triggering automatic pipeline re-execution from point of mutation...")
        try:
            sync_status = self.tracker.invalidate_and_sync(auto_rerun=True)
            logger.info("Pipeline synchronization completed successfully.")
            return sync_status
        except Exception as exc:
            logger.error("Pipeline re-synchronization aborted: %s", exc)
            self._abort(f"Provenance synchronization failed: {exc}")
            raise

    def run_automated_test_suite(self) -> Dict[str, Any]:
        """Executes pytest test suite collecting line and branch coverage metrics.

        Returns:
            Dictionary containing test execution counts, durations, and coverage rates.
        """
        logger.info("=" * 70)
        logger.info("[PHASE 2/4] EXECUTING RIGOROUS AUTOMATED TEST SUITE & COVERAGE AUDIT...")
        logger.info("=" * 70)

        coverage_json_path = self.paths.reports_dir / "coverage.json"
        coverage_json_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            f"--cov=src.core",
            f"--cov=src.data",
            f"--cov-report=json:{coverage_json_path}",
            "--cov-report=term",
            "--cov-branch",
            "--cov-config=.coveragerc",
        ]

        start_time = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=str(self.paths.root_dir),
            capture_output=True,
            text=True,
        )
        duration = time.perf_counter() - start_time

        logger.info("Pytest stdout:\n%s", result.stdout)
        if result.stderr:
            logger.warning("Pytest stderr:\n%s", result.stderr)

        if result.returncode != 0:
            logger.error("Test Suite Execution Failed! Aborting pre-flight gate.")
            self._abort(f"Pytest exited with non-zero code {result.returncode}.")

        # Parse coverage metrics from JSON report
        if not coverage_json_path.exists():
            self._abort(f"Coverage JSON report missing at: {coverage_json_path}")

        with open(coverage_json_path, "r", encoding="utf-8") as f:
            cov_data = json.load(f)

        totals = cov_data.get("totals", {})
        percent_covered = float(totals.get("percent_covered", 0.0))
        num_statements = int(totals.get("num_statements", 0))
        missing_lines = int(totals.get("missing_lines", 0))
        num_branches = int(totals.get("num_branches", 0))
        missing_branches = int(totals.get("missing_branches", 0))

        logger.info(
            "Coverage Metrics: %.2f%% total coverage (Target: >=%.2f%%)",
            percent_covered,
            self.coverage_threshold,
        )
        logger.info(
            "Statements: %d total, %d missing | Branches: %d total, %d partial",
            num_statements,
            missing_lines,
            num_branches,
            missing_branches,
        )

        if percent_covered < self.coverage_threshold:
            self._abort(
                f"Coverage {percent_covered:.2f}% falls below designated {self.coverage_threshold:.2f}% threshold!"
            )

        return {
            "exit_code": result.returncode,
            "duration_seconds": round(duration, 2),
            "percent_covered": percent_covered,
            "num_statements": num_statements,
            "missing_lines": missing_lines,
            "num_branches": num_branches,
            "missing_branches": missing_branches,
            "status": "PASSED",
        }

    def audit_dataset_distribution(self) -> Dict[str, Any]:
        """Audits live dataset partitions, class balances, and bounding box counts.

        Returns:
            Dictionary containing active sample allocations and split statistics.
        """
        logger.info("=" * 70)
        logger.info("[PHASE 3/4] AUDITING DATASET DISTRIBUTION AND CLASS STRATIFICATION...")
        logger.info("=" * 70)

        manifest = pd.read_csv(self.paths.manifest_path)
        active_df = manifest[manifest["is_pruned"] == False].copy().reset_index(drop=True)

        total_images = len(active_df)
        train_df = active_df[active_df["split"] == "train"]
        val_df = active_df[active_df["split"] == "val"]
        test_df = active_df[active_df["split"] == "test"]

        # Audit bounding boxes from master labels
        total_boxes = 0
        boxes_per_class: Dict[str, int] = {}
        for fname in active_df["filename"]:
            lbl_file = self.paths.master_labels_dir / fname.replace(".jpg", ".txt").replace(".png", ".txt")
            if lbl_file.exists():
                lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
                total_boxes += len(lines)

        classes = sorted(list(active_df["class_name"].unique()))
        table_rows = []
        class_summary: Dict[str, Dict[str, Any]] = {}

        for c in classes:
            c_df = active_df[active_df["class_name"] == c]
            tr_c = len(c_df[c_df["split"] == "train"])
            va_c = len(c_df[c_df["split"] == "val"])
            te_c = len(c_df[c_df["split"] == "test"])
            tot_c = len(c_df)

            tr_pct = f"{tr_c / tot_c * 100:.1f}%"
            va_pct = f"{va_c / tot_c * 100:.1f}%"
            te_pct = f"{te_c / tot_c * 100:.1f}%"

            table_rows.append([c, tot_c, f"{tr_c} ({tr_pct})", f"{va_c} ({va_pct})", f"{te_c} ({te_pct})"])
            class_summary[c] = {
                "total": tot_c,
                "train": tr_c,
                "val": va_c,
                "test": te_c,
            }

        headers = ["Denomination", "Total Images", "Train Count", "Val Count", "Test Count"]
        logger.info("\n%s", format_table(headers, table_rows))

        # Split bounding boxes count
        split_boxes: Dict[str, int] = {}
        for s in ["train", "val", "test"]:
            s_box_count = 0
            for lbl_file in (self.paths.labels_dir / s).glob("*.txt"):
                content = lbl_file.read_text(encoding="utf-8").strip()
                if content:
                    s_box_count += len(content.splitlines())
            split_boxes[s] = s_box_count

        logger.info(
            "Split Box Totals: Train=%d, Val=%d, Test=%d | Total Annotations=%d",
            split_boxes.get("train", 0),
            split_boxes.get("val", 0),
            split_boxes.get("test", 0),
            total_boxes,
        )

        return {
            "active_image_count": total_images,
            "total_bounding_boxes": total_boxes,
            "split_image_counts": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "split_box_counts": split_boxes,
            "class_distribution": class_summary,
        }

    def generate_verification_report(
        self,
        prov_status: InvalidationStatus,
        test_summary: Dict[str, Any],
        dist_summary: Dict[str, Any],
    ) -> Path:
        """Serializes pre-flight audit report to disk.

        Args:
            prov_status: Status output from provenance inspection.
            test_summary: Summary metrics from test suite and coverage runner.
            dist_summary: Summary metrics from dataset distribution audit.

        Returns:
            Path pointing to written JSON report.
        """
        logger.info("=" * 70)
        logger.info("[PHASE 4/4] GENERATING PRE-FLIGHT VERIFICATION REPORT...")
        logger.info("=" * 70)

        # Read latest split summary if available
        split_metrics = {}
        if self.paths.split_summary_path.exists():
            with open(self.paths.split_summary_path, "r", encoding="utf-8") as f:
                split_metrics = json.load(f)

        cached_manifest = self.tracker.load_state_manifest() or {}
        provenance_digests = {k: v.combined_hash for k, v in cached_manifest.items()}

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gate_status": "APPROVED_FOR_TRAINING",
            "coverage_mandate_met": test_summary["percent_covered"] >= self.coverage_threshold,
            "test_summary": test_summary,
            "dataset_summary": dist_summary,
            "zero_leakage_metrics": split_metrics,
            "provenance_fingerprints": provenance_digests,
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Pre-flight verification report saved to: %s", self.report_path)
        return self.report_path

    def _abort(self, reason: str) -> None:
        """Logs critical failure and exits process with non-zero exit code.

        Args:
            reason: Diagnostic explanation of the gatekeeper failure.
        """
        logger.critical("PRE-FLIGHT GATE REJECTED: %s", reason)
        logger.critical("Compute allocation blocked. Fix reported violations before training.")
        sys.exit(1)

    def execute_gate(self) -> None:
        """Orchestrates end-to-end pre-flight gating procedure."""
        print("\n" + "=" * 70)
        print("  AZERBAIJANI BANKNOTE DETECTION: PRE-TRAINING QUALITY GATE")
        print("  Enterprise Gatekeeper & Pipeline Invalidation Protocol")
        print("=" * 70 + "\n")

        prov_status = self.run_provenance_check()
        test_summary = self.run_automated_test_suite()
        dist_summary = self.audit_dataset_distribution()
        report_file = self.generate_verification_report(prov_status, test_summary, dist_summary)

        print("\n" + "#" * 70)
        print("  PRE-FLIGHT GATE VERDICT: PASSED (ALL CRITERIA SATISFIED)")
        print(f"  Coverage: {test_summary['percent_covered']:.2f}% (Target: >={self.coverage_threshold:.2f}%)")
        print(f"  Active Dataset: {dist_summary['active_image_count']} images, {dist_summary['total_bounding_boxes']} boxes")
        print(f"  Audit Report: {report_file}")
        print("  STATUS: EXECUTION RELEASED TO TRAINING PIPELINE.")
        print("#" * 70 + "\n")


if __name__ == "__main__":
    gatekeeper = PreFlightGatekeeper()
    gatekeeper.execute_gate()
