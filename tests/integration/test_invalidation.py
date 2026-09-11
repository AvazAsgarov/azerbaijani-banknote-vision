"""Integration Tests for Mandatory Pipeline Invalidation and Staleness Detection.

Evaluates dependency DAG propagation when upstream raw data, annotations,
configurations, or pipeline code mutate prior to training execution.
"""

from pathlib import Path
import pytest

from src.core.config import ProjectPaths
from src.core.provenance import InvalidationStatus, ProvenanceTracker


@pytest.mark.integration
class TestPipelineInvalidationFlow:
    """Evaluates DAG invalidation triggers across end-to-end pipeline stages."""

    def test_invalidation_lifecycle_on_mutation(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies full lifecycle of state capture, mutation detection, and invalidation."""
        tracker = ProvenanceTracker(populated_mock_env)

        # Baseline capture
        initial_state = tracker.capture_current_state()
        tracker.save_state_manifest(initial_state)

        # Confirm baseline is fresh
        status1 = tracker.evaluate_invalidation()
        assert status1.is_fresh is True
        assert len(status1.stale_stages) == 0

        # Inject mutation: modify an annotation in master_labels
        target_label = next(populated_mock_env.master_labels_dir.glob("*.txt"))
        original_text = target_label.read_text(encoding="utf-8")
        target_label.write_text(original_text + "\n1 0.2 0.2 0.3 0.3\n", encoding="utf-8")

        # Evaluate invalidation
        status2 = tracker.evaluate_invalidation()
        assert status2.is_fresh is False
        assert "master_labels" in status2.stale_stages
        # Verify downstream stages are invalidated via DAG
        assert "embeddings" in status2.stale_stages
        assert "splits" in status2.stale_stages
        assert "reports" in status2.stale_stages

        # Re-capture and save to simulate post-execution sync
        updated_state = tracker.capture_current_state()
        tracker.save_state_manifest(updated_state)

        # Confirm state returns to fresh
        status3 = tracker.evaluate_invalidation()
        assert status3.is_fresh is True
        assert len(status3.stale_stages) == 0

    def test_config_mutation_invalidates_all_downstream(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies configuration mutations invalidate embeddings, splits, and reports."""
        tracker = ProvenanceTracker(populated_mock_env)
        state = tracker.capture_current_state()
        tracker.save_state_manifest(state)

        # Mutate configuration YAML
        cfg_file = populated_mock_env.data_yaml_path
        cfg_file.write_text("names:\n  0: 001_azn\n  1: 005_azn\nnc: 2\n", encoding="utf-8")

        status = tracker.evaluate_invalidation()
        assert status.is_fresh is False
        assert "configs" in status.stale_stages
        assert "embeddings" in status.stale_stages
        assert "splits" in status.stale_stages
        assert "reports" in status.stale_stages
