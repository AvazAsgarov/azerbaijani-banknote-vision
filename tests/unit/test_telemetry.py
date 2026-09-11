"""
Unit tests for MLOps training telemetry and dashboard callback mechanisms.
Verifies event logging, duration tracking, state persistence, and model attachment.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path

from src.telemetry.telemetry_hook import DashboardTelemetryCallback, attach_telemetry_callbacks


@pytest.mark.unit
class TestDashboardTelemetryCallback:
    """Test suite verifying DashboardTelemetryCallback event tracking."""

    def test_callback_initialization_creates_target_directory(self):
        """Verify initialization creates target directory and initializes tracking variables.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "telemetry_sub"
            cb = DashboardTelemetryCallback(output_dir=out_path, experiment_name="test_exp")

            assert out_path.exists()
            assert cb.experiment_name == "test_exp"
            assert cb.batch_count == 0

    def test_on_train_start_logs_metadata_record(self):
        """Verify train start event records experiment metadata and creates stream files.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            cb = DashboardTelemetryCallback(output_dir=out_path, experiment_name="unit_test_exp")

            class MockTrainer:
                epochs = 25
                batch_size = 16

            cb.on_train_start(MockTrainer())

            assert cb.jsonl_path.exists()
            assert cb.state_path.exists()

            with open(cb.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            assert state["event"] == "train_start"
            assert state["epochs"] == 25
            assert state["batch_size"] == 16

    def test_epoch_lifecycle_metrics_tracking(self):
        """Verify epoch lifecycle methods track duration and batch counters.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            cb = DashboardTelemetryCallback(output_dir=out_path)
            cb.on_train_start(None)
            cb.on_train_epoch_start(None)
            cb.on_train_batch_end(None)
            cb.on_train_batch_end(None)

            class MockTrainer:
                epoch = 0
                loss_items = [0.45, 0.32]
                metrics = {"mAP50": 0.88}

            cb.on_train_epoch_end(MockTrainer())

            with open(cb.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            assert state["event"] == "epoch_end"
            assert state["epoch"] == 1
            assert state["batch_count"] == 2
            assert state["loss_items"]["loss_0"] == 0.45
            assert state["metrics"]["mAP50"] == 0.88

    def test_on_train_end_finalizes_duration(self):
        """Verify train end event persists cumulative duration records.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            cb = DashboardTelemetryCallback(output_dir=out_path)
            cb.on_train_start(None)
            cb.on_train_end(None)

            with open(cb.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            assert state["event"] == "train_end"
            assert "total_duration_sec" in state

    def test_attach_telemetry_callbacks_registers_handlers(self):
        """Verify helper function attaches callbacks when model provides add_callback method.

        Args:
            None.

        Returns:
            None.
        """
        registered_events = {}

        class MockModel:
            def add_callback(self, event, handler):
                registered_events[event] = handler

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = attach_telemetry_callbacks(MockModel(), Path(tmpdir))

            assert isinstance(cb, DashboardTelemetryCallback)
            assert "on_train_start" in registered_events
            assert "on_train_epoch_end" in registered_events
            assert "on_train_end" in registered_events
