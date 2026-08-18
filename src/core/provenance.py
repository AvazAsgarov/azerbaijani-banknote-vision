"""Cryptographic Provenance and Invalidation Engine for Dataset Preparation.

Deterministic SHA-256 state signatures track mutations across source images,
annotations, configurations, and pipeline code. A directed dependency graph
evaluates artifact staleness, triggering targeted downstream re-executions
when upstream representations or hyperparameters mutate.
"""

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.core.config import ProjectPaths

logger = logging.getLogger("ProvenanceEngine")


@dataclass
class StageSignature:
    """Cryptographic fingerprints representing a single pipeline stage.

    Attributes:
        stage_name: Unique identifier for the pipeline processing stage.
        combined_hash: Merged SHA-256 hex digest representing all constituent elements.
        file_hashes: Mapping from relative file paths to individual SHA-256 digests.
        timestamp: ISO 8601 UTC timestamp recording fingerprint generation.
        file_count: Total tracked files incorporated into the signature.
    """

    stage_name: str
    combined_hash: str
    file_hashes: Dict[str, str]
    timestamp: str
    file_count: int


@dataclass
class InvalidationStatus:
    """Diagnostic assessment of pipeline freshness across all stages.

    Attributes:
        is_fresh: Confirmation that all downstream artifacts match upstream inputs.
        stale_stages: Ordered list of processing stages requiring re-execution.
        mutated_files: Mapping of stage identifiers to lists of changed file paths.
        reasons: Descriptive summary of detected mutations or missing artifacts.
    """

    is_fresh: bool
    stale_stages: List[str]
    mutated_files: Dict[str, List[str]]
    reasons: List[str]


class ProvenanceTracker:
    """Manages cryptographic state hashing and enforces artifact dependency DAGs.

    Attributes:
        paths: Centralized filesystem paths container.
        manifest_path: Filesystem path to the stored provenance state JSON.
    """

    def __init__(self, paths: Optional[ProjectPaths] = None) -> None:
        """Initializes provenance tracker with project directory paths.

        Args:
            paths: Filesystem path configuration container.
        """
        self.paths = paths or ProjectPaths()
        self.manifest_path: Path = self.paths.artifacts_dir / "provenance_manifest.json"

    @staticmethod
    def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
        """Calculates SHA-256 hexadecimal digest for a single filesystem file.

        Args:
            file_path: Target path pointing to the file.
            chunk_size: Stream buffer size in bytes for chunked reading.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def compute_directory_signature(
        self,
        stage_name: str,
        directory_path: Path,
        glob_patterns: List[str],
        exclude_patterns: Optional[List[str]] = None,
    ) -> StageSignature:
        """Computes a combined cryptographic signature for target directory contents.

        Files matching glob patterns are sorted lexicographically by relative path.
        Individual SHA-256 digests are concatenated to generate a deterministic
        aggregate stage fingerprint.

        Args:
            stage_name: Logical identifier for the pipeline stage.
            directory_path: Base directory containing target files.
            glob_patterns: List of glob expressions (e.g., ['*.py', '*.txt']).
            exclude_patterns: Optional patterns designating files for exclusion.

        Returns:
            StageSignature containing individual and combined SHA-256 digests.
        """
        file_hashes: Dict[str, str] = {}
        if not directory_path.exists():
            return StageSignature(
                stage_name=stage_name,
                combined_hash="",
                file_hashes={},
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_count=0,
            )

        matched_files: Set[Path] = set()
        for pattern in glob_patterns:
            matched_files.update(directory_path.glob(pattern))

        excludes: List[str] = exclude_patterns or []
        filtered_files: List[Path] = []
        for file_path in matched_files:
            if not file_path.is_file():
                continue
            relative_str = str(file_path.relative_to(self.paths.root_dir)).replace("\\", "/")
            if any(ex in relative_str for ex in excludes):
                continue
            filtered_files.append(file_path)

        filtered_files.sort(key=lambda p: str(p.relative_to(self.paths.root_dir)).replace("\\", "/"))

        aggregate_hasher = hashlib.sha256()
        for file_path in filtered_files:
            rel_path = str(file_path.relative_to(self.paths.root_dir)).replace("\\", "/")
            digest = self.compute_file_hash(file_path)
            file_hashes[rel_path] = digest
            aggregate_hasher.update(f"{rel_path}:{digest}".encode("utf-8"))

        combined_digest = aggregate_hasher.hexdigest() if file_hashes else ""
        return StageSignature(
            stage_name=stage_name,
            combined_hash=combined_digest,
            file_hashes=file_hashes,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_count=len(file_hashes),
        )

    def capture_current_state(self) -> Dict[str, StageSignature]:
        """Calculates current signatures across all upstream and downstream stages.

        Returns:
            Dictionary mapping stage identifiers to their respective StageSignature.
        """
        signatures: Dict[str, StageSignature] = {}

        signatures["configs"] = self.compute_directory_signature(
            stage_name="configs",
            directory_path=self.paths.configs_dir,
            glob_patterns=["*.yaml", "*.yml"],
        )

        signatures["pipeline_code"] = self.compute_directory_signature(
            stage_name="pipeline_code",
            directory_path=self.paths.root_dir / "src",
            glob_patterns=["**/*.py"],
            exclude_patterns=["__pycache__"],
        )

        signatures["master_labels"] = self.compute_directory_signature(
            stage_name="master_labels",
            directory_path=self.paths.master_labels_dir,
            glob_patterns=["*.txt"],
        )

        signatures["master_images"] = self.compute_directory_signature(
            stage_name="master_images",
            directory_path=self.paths.master_images_dir,
            glob_patterns=["*.jpg", "*.jpeg", "*.png"],
        )

        signatures["embeddings"] = self.compute_directory_signature(
            stage_name="embeddings",
            directory_path=self.paths.embeddings_dir,
            glob_patterns=["*.h5", "*.parquet", "*.npy"],
        )

        signatures["splits"] = self.compute_directory_signature(
            stage_name="splits",
            directory_path=self.paths.labels_dir,
            glob_patterns=["**/*.txt"],
        )

        signatures["reports"] = self.compute_directory_signature(
            stage_name="reports",
            directory_path=self.paths.reports_dir,
            glob_patterns=["*.json"],
        )

        return signatures

    def save_state_manifest(self, signatures: Dict[str, StageSignature]) -> Path:
        """Serializes captured stage signatures to JSON provenance manifest on disk.

        Args:
            signatures: Mapping from stage names to computed signatures.

        Returns:
            Path pointing to the written manifest.
        """
        payload = {
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "stages": {name: asdict(sig) for name, sig in signatures.items()},
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Provenance state manifest saved to %s", self.manifest_path)
        return self.manifest_path

    def load_state_manifest(self) -> Optional[Dict[str, StageSignature]]:
        """Loads previously saved provenance signatures from disk.

        Returns:
            Dictionary of StageSignature objects, or None when manifest is absent or invalid.
        """
        if not self.manifest_path.exists():
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stages_data = data.get("stages", {})
            return {
                name: StageSignature(
                    stage_name=info["stage_name"],
                    combined_hash=info["combined_hash"],
                    file_hashes=info["file_hashes"],
                    timestamp=info["timestamp"],
                    file_count=info["file_count"],
                )
                for name, info in stages_data.items()
            }
        except Exception as exc:
            logger.warning("Manifest decoding failure at %s: %s", self.manifest_path, exc)
            return None

    def evaluate_invalidation(self) -> InvalidationStatus:
        """Evaluates staleness across the artifact dependency DAG.

        A stage is flagged as stale if its stored signature differs from the live
        state, or if any upstream dependency on which it relies has mutated.

        Returns:
            InvalidationStatus detailing freshness, stale stages, and reasons.
        """
        current = self.capture_current_state()
        cached = self.load_state_manifest()

        stale_stages: List[str] = []
        mutated_files: Dict[str, List[str]] = {}
        reasons: List[str] = []

        if cached is None:
            reasons.append("Initial execution: no cached provenance manifest discovered.")
            return InvalidationStatus(
                is_fresh=False,
                stale_stages=list(current.keys()),
                mutated_files={k: list(v.file_hashes.keys()) for k, v in current.items()},
                reasons=reasons,
            )

        # Direct mutation detection per stage
        for stage_name, curr_sig in current.items():
            prev_sig = cached.get(stage_name)
            if prev_sig is None:
                stale_stages.append(stage_name)
                reasons.append(f"Stage '{stage_name}' has no prior signature.")
                continue

            if curr_sig.combined_hash != prev_sig.combined_hash:
                changed = []
                for p, h in curr_sig.file_hashes.items():
                    if prev_sig.file_hashes.get(p) != h:
                        changed.append(p)
                for p in prev_sig.file_hashes:
                    if p not in curr_sig.file_hashes:
                        changed.append(f"[DELETED] {p}")

                stale_stages.append(stage_name)
                mutated_files[stage_name] = changed
                reasons.append(f"Stage '{stage_name}' mutated: {len(changed)} file(s) changed.")

        # Upstream DAG dependency propagation
        # Rule 1: Configs / Master Data mutation forces Embeddings invalidation
        if any(s in stale_stages for s in ["configs", "master_images", "master_labels"]):
            if "embeddings" not in stale_stages:
                stale_stages.append("embeddings")
                reasons.append("Embeddings invalidated due to upstream master data/config mutation.")

        # Rule 2: Embeddings or Configs mutation forces Splits invalidation
        if any(s in stale_stages for s in ["embeddings", "configs"]):
            if "splits" not in stale_stages:
                stale_stages.append("splits")
                reasons.append("Splits invalidated due to upstream embeddings/config mutation.")

        # Rule 3: Splits mutation forces Reports/EDA invalidation
        if "splits" in stale_stages:
            if "reports" not in stale_stages:
                stale_stages.append("reports")
                reasons.append("Reports invalidated due to split modifications.")

        is_fresh = len(stale_stages) == 0
        return InvalidationStatus(
            is_fresh=is_fresh,
            stale_stages=stale_stages,
            mutated_files=mutated_files,
            reasons=reasons,
        )

    def invalidate_and_sync(self, auto_rerun: bool = True) -> InvalidationStatus:
        """Executes targeted re-execution of stale stages to restore pipeline freshness.

        Args:
            auto_rerun: When true, triggers sub-pipeline execution commands.

        Returns:
            InvalidationStatus summarizing the post-synchronization state.
        """
        status = self.evaluate_invalidation()
        if status.is_fresh:
            logger.info("Pipeline artifacts confirm complete freshness. Zero mutations detected.")
            return status

        logger.warning(
            "Pipeline staleness detected across %d stages: %s",
            len(status.stale_stages),
            ", ".join(status.stale_stages),
        )

        if not auto_rerun:
            return status

        # Trigger downstream execution according to DAG order
        if "splits" in status.stale_stages:
            logger.info("Triggering zero-leakage split re-execution...")
            split_script = self.paths.root_dir / "scripts" / "run_zero_leakage_split.py"
            result = subprocess.run(
                [sys.executable, str(split_script)],
                cwd=str(self.paths.root_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Split re-execution failed: %s", result.stderr)
                raise RuntimeError(f"Split re-execution aborted with code {result.returncode}")

        if "reports" in status.stale_stages:
            logger.info("Triggering advanced EDA and diagnostic regeneration...")
            eda_script = self.paths.root_dir / "scripts" / "run_advanced_eda.py"
            result = subprocess.run(
                [sys.executable, str(eda_script)],
                cwd=str(self.paths.root_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("EDA re-execution failed: %s", result.stderr)
                raise RuntimeError(f"EDA regeneration aborted with code {result.returncode}")

        # Capture and persist fresh state upon successful execution
        fresh_state = self.capture_current_state()
        self.save_state_manifest(fresh_state)
        logger.info("Pipeline successfully synchronized. State manifest updated.")
        return InvalidationStatus(
            is_fresh=True,
            stale_stages=[],
            mutated_files={},
            reasons=["Pipeline successfully synchronized."],
        )
