"""Unit Tests for Provenance and Pipeline Invalidation Engine.

Verifies SHA-256 fingerprint generation, artifact state persistence,
DAG-based staleness propagation, and automatic re-execution triggers.
"""

import json
from pathlib import Path
import pytest

from src.core.config import ProjectPaths
from src.core.provenance import InvalidationStatus, ProvenanceTracker, StageSignature


@pytest.mark.unit
class TestProvenanceTracker:
    """Tests cryptographic state hashing and invalidation detection."""

    def test_file_hash_calculation(self, tmp_path: Path) -> None:
        """Verifies deterministic SHA-256 computation for files."""
        test_file = tmp_path / "sample.txt"
        test_file.write_text("AZN Banknote Detection", encoding="utf-8")
        digest1 = ProvenanceTracker.compute_file_hash(test_file)
        digest2 = ProvenanceTracker.compute_file_hash(test_file)
        assert digest1 == digest2
        assert len(digest1) == 64

    def test_empty_file_hash(self, tmp_path: Path) -> None:
        """Verifies SHA-256 calculation for zero-byte files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()
        digest = ProvenanceTracker.compute_file_hash(empty_file)
        # Canonical SHA-256 hash of empty byte string
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_directory_signature_nonexistent(self, mock_paths: ProjectPaths) -> None:
        """Verifies signature generation handles missing directories gracefully."""
        tracker = ProvenanceTracker(mock_paths)
        missing_dir = mock_paths.root_dir / "non_existent_folder"
        sig = tracker.compute_directory_signature("test_stage", missing_dir, ["*.txt"])
        assert sig.combined_hash == ""
        assert sig.file_count == 0
        assert sig.file_hashes == {}

    def test_directory_signature_with_exclusions(self, mock_paths: ProjectPaths) -> None:
        """Verifies glob matches and pattern exclusions."""
        tracker = ProvenanceTracker(mock_paths)
        test_dir = mock_paths.root_dir / "test_dir"
        test_dir.mkdir(parents=True)
        (test_dir / "valid.txt").write_text("valid", encoding="utf-8")
        (test_dir / "cache.tmp").write_text("temp", encoding="utf-8")

        sig = tracker.compute_directory_signature(
            stage_name="custom_stage",
            directory_path=test_dir,
            glob_patterns=["*.*"],
            exclude_patterns=[".tmp"],
        )
        assert sig.file_count == 1
        assert any("valid.txt" in k for k in sig.file_hashes)
        assert not any("cache.tmp" in k for k in sig.file_hashes)

    def test_save_and_load_manifest(self, mock_paths: ProjectPaths) -> None:
        """Verifies serialization and deserialization of provenance manifest."""
        tracker = ProvenanceTracker(mock_paths)
        state = tracker.capture_current_state()
        manifest_file = tracker.save_state_manifest(state)
        assert manifest_file.exists()

        loaded = tracker.load_state_manifest()
        assert loaded is not None
        assert "configs" in loaded
        assert loaded["configs"].combined_hash == state["configs"].combined_hash

    def test_load_manifest_corrupted_json(self, mock_paths: ProjectPaths) -> None:
        """Verifies graceful recovery when manifest contains invalid JSON."""
        tracker = ProvenanceTracker(mock_paths)
        tracker.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tracker.manifest_path.write_text("{corrupted_json_content", encoding="utf-8")
        loaded = tracker.load_state_manifest()
        assert loaded is None

    def test_invalidation_initial_state(self, mock_paths: ProjectPaths) -> None:
        """Verifies initial un-manifested environment is flagged as stale."""
        tracker = ProvenanceTracker(mock_paths)
        if tracker.manifest_path.exists():
            tracker.manifest_path.unlink()
        status = tracker.evaluate_invalidation()
        assert status.is_fresh is False
        assert len(status.stale_stages) > 0

    def test_invalidation_clean_state(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies freshness confirmation when filesystem matches stored manifest."""
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        tracker.save_state_manifest(state)

        status = tracker.evaluate_invalidation()
        assert status.is_fresh is True
        assert status.stale_stages == []
        assert status.mutated_files == {}

    def test_invalidation_upon_label_mutation(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies label file modification triggers downstream DAG invalidation."""
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        tracker.save_state_manifest(state)

        # Mutate an annotation file
        label_file = next(populated_mock_env.master_labels_dir.glob("*.txt"))
        original_content = label_file.read_text(encoding="utf-8")
        label_file.write_text(original_content + "\n0 0.1 0.1 0.2 0.2\n", encoding="utf-8")

        status = tracker.evaluate_invalidation()
        assert status.is_fresh is False
        assert "master_labels" in status.stale_stages
        # Verify DAG propagation: master label change invalidates embeddings, splits, reports
        assert "embeddings" in status.stale_stages
        assert "splits" in status.stale_stages
        assert "reports" in status.stale_stages

    def test_invalidation_upon_file_deletion(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies file deletion triggers staleness detection."""
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        tracker.save_state_manifest(state)

        # Delete an image file
        target_img = next(populated_mock_env.master_images_dir.glob("*.jpg"))
        target_img.unlink()

        status = tracker.evaluate_invalidation()
        assert status.is_fresh is False
        assert "master_images" in status.stale_stages
        assert any("[DELETED]" in item for item in status.mutated_files["master_images"])

    def test_invalidate_and_sync_already_fresh(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies synchronization terminates early when state is already fresh."""
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        tracker.save_state_manifest(state)

        status = tracker.invalidate_and_sync(auto_rerun=False)
        assert status.is_fresh is True

    def test_invalidate_and_sync_no_auto_rerun(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies auto_rerun=False returns status without running subprocesses."""
        tracker = ProvenanceTracker(populated_mock_env)
        # Create un-synced mutation
        (populated_mock_env.configs_dir / "data.yaml").write_text("names: [0]\n", encoding="utf-8")
        status = tracker.invalidate_and_sync(auto_rerun=False)
        assert status.is_fresh is False
        assert len(status.stale_stages) > 0

    def test_invalidation_missing_stage_in_cached(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies evaluation detects when cached manifest lacks an existing stage."""
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        # Remove a stage from cached signatures
        del state["reports"]
        tracker.save_state_manifest(state)

        status = tracker.evaluate_invalidation()
        assert status.is_fresh is False
        assert "reports" in status.stale_stages

    def test_invalidate_and_sync_auto_rerun_success(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies auto_rerun triggers subprocesses and updates manifest upon success."""
        from unittest.mock import MagicMock, patch
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        tracker.save_state_manifest(state)

        # Mutate label to flag splits and reports
        target_label = next(populated_mock_env.master_labels_dir.glob("*.txt"))
        target_label.write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            status = tracker.invalidate_and_sync(auto_rerun=True)
            assert status.is_fresh is True
            assert mock_run.call_count >= 1

    def test_invalidate_and_sync_auto_rerun_failure(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies auto_rerun raises RuntimeError when subprocess execution fails."""
        from unittest.mock import MagicMock, patch
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        tracker.save_state_manifest(state)

        # Mutate label to trigger invalidation
        target_label = next(populated_mock_env.master_labels_dir.glob("*.txt"))
        target_label.write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Process error"
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError):
                tracker.invalidate_and_sync(auto_rerun=True)

