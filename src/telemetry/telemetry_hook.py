"""
Standardized Training Telemetry Hook & Callback Module.

Provides decoupled, high-resolution callbacks for Ultralytics YOLO/RT-DETR
and PyTorch training loops to record microsecond timestamps, hardware metrics,
and epoch telemetry into local JSONL streams.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("TelemetryHook")


class DashboardTelemetryCallback:
    """Ultralytics and PyTorch lifecycle callback tracking telemetry and epoch durations."""

    def __init__(self, output_dir: Path, experiment_name: str = "exp1_architecture_battle") -> None:
        """Initializes callback with stream output directory and experiment metadata.

        Args:
            output_dir: Directory where JSONL streams and state snapshots are persisted.
            experiment_name: Descriptive identifier tag for the active training experiment.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.jsonl_path = self.output_dir / "telemetry_stream.jsonl"
        self.state_path = self.output_dir / "telemetry_current_state.json"
        self.train_start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.batch_count = 0

    def _append_log(self, record: Dict[str, Any]) -> None:
        """Appends a structured JSON record into the telemetry JSONL stream."""
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("Failed writing telemetry record: %s", exc)

    def _write_state(self, state: Dict[str, Any]) -> None:
        """Persists the latest snapshot state atomically."""
        try:
            temp_path = self.state_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            temp_path.replace(self.state_path)
        except Exception as exc:
            logger.debug("Failed updating telemetry state: %s", exc)

    def on_train_start(self, trainer: Any) -> None:
        """Records initial environment parameters and run metadata."""
        self.train_start_time = time.time()
        record = {
            "event": "train_start",
            "experiment": self.experiment_name,
            "timestamp": self.train_start_time,
            "epochs": getattr(trainer, "epochs", None),
            "batch_size": getattr(trainer, "batch_size", None),
        }
        self._append_log(record)
        self._write_state(record)

    def on_train_epoch_start(self, trainer: Any) -> None:
        """Marks epoch start time to calculate exact training duration."""
        self.epoch_start_time = time.time()
        self.batch_count = 0

    def on_train_batch_end(self, trainer: Any) -> None:
        """Tracks batch iterations and updates internal batch counter.

        Args:
            trainer: Active trainer instance dispatching the callback event.
        """
        self.batch_count += 1

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Calculates epoch duration and records validation and training metrics."""
        now = time.time()
        epoch_duration = round(now - (self.epoch_start_time or now), 2)
        total_elapsed = round(now - (self.train_start_time or now), 2)
        epoch = getattr(trainer, "epoch", 0) + 1

        # Extract loss and metrics dictionaries if present
        loss_items = {}
        if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
            loss_items = {f"loss_{i}": float(v) for i, v in enumerate(trainer.loss_items)}

        metrics = {}
        if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
            metrics = {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (int, float))}

        record = {
            "event": "epoch_end",
            "epoch": epoch,
            "epoch_duration_sec": epoch_duration,
            "total_elapsed_sec": total_elapsed,
            "batch_count": self.batch_count,
            "loss_items": loss_items,
            "metrics": metrics,
            "timestamp": now,
        }
        self._append_log(record)
        self._write_state(record)

    def on_train_end(self, trainer: Any) -> None:
        """Marks training completion and logs total cumulative duration."""
        now = time.time()
        total_duration = round(now - (self.train_start_time or now), 2)
        record = {
            "event": "train_end",
            "total_duration_sec": total_duration,
            "total_duration_minutes": round(total_duration / 60.0, 2),
            "timestamp": now,
        }
        self._append_log(record)
        self._write_state(record)


def attach_telemetry_callbacks(model: Any, output_dir: Path, experiment_name: str = "exp1_architecture_battle") -> DashboardTelemetryCallback:
    """Attaches standard telemetry callbacks to an Ultralytics YOLO or RTDETR model instance.

    Args:
        model: Ultralytics YOLO or RTDETR object.
        output_dir: Local or remote directory storing telemetry stream files.
        experiment_name: Identifier tag for this experiment phase.

    Returns:
        The configured DashboardTelemetryCallback instance.
    """
    callback = DashboardTelemetryCallback(output_dir=output_dir, experiment_name=experiment_name)
    if hasattr(model, "add_callback"):
        model.add_callback("on_train_start", callback.on_train_start)
        model.add_callback("on_train_epoch_start", callback.on_train_epoch_start)
        model.add_callback("on_train_batch_end", callback.on_train_batch_end)
        model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
        model.add_callback("on_train_end", callback.on_train_end)
        logger.info("Attached DashboardTelemetryCallback to model successfully.")
    return callback
