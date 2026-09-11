"""Experiment Data Contracts, Configurations, and Metric Schemas.

Defines standardized data models for orchestrating, executing, validating,
and archiving experiments 1 through 6 on Azerbaijani Banknote detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ArmConfig:
    """Configuration contract for a specific model arm or ablation variation.

    Attributes:
        arm_id: Machine-readable unique identifier (e.g. 'yolo11m', 'arm2_geometric').
        display_name: Publication-ready descriptive label.
        model_architecture: Neural architecture identifier.
        weights: Initial checkpoint or backbone identifier.
        hparams: Hyperparameter dictionary passed to training engine.
        description: Brief scientific explanation of this arm's ablation role.
    """

    arm_id: str
    display_name: str
    model_architecture: str
    weights: str = "yolo11m.pt"
    hparams: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class RunMetrics:
    """Quantitative evaluation and telemetry metrics for a completed arm run.

    Attributes:
        arm_id: Identifier matching the arm configuration.
        train_duration_sec: Total elapsed wall-clock training time.
        epochs_completed: Number of epochs trained before early stopping or target.
        val_map50: Mean Average Precision at IoU=0.50 on validation split.
        val_map50_95: Mean Average Precision at IoU=0.50:0.95 on validation split.
        val_precision: Mean precision across classes on validation split.
        val_recall: Mean recall across classes on validation split.
        test_map50: Mean Average Precision at IoU=0.50 on zero-leakage test split.
        test_map50_95: Mean Average Precision at IoU=0.50:0.95 on zero-leakage test split.
        test_precision: Mean precision across classes on test split.
        test_recall: Mean recall across classes on test split.
        parameters_m: Trainable parameter count in millions.
        gflops: Computation complexity in GFLOPs at nominal resolution.
        latency_ms: Single-image forward inference latency in milliseconds.
        fps: Frames per second throughput on target hardware.
        model_size_mb: Physical disk footprint of serialized model checkpoint.
        per_class_test_ap: Denomination-level breakdown of test AP50 and AP50-95.
    """

    arm_id: str
    train_duration_sec: float = 0.0
    epochs_completed: int = 0
    val_map50: float = 0.0
    val_map50_95: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    test_map50: float = 0.0
    test_map50_95: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    parameters_m: float = 0.0
    gflops: float = 0.0
    latency_ms: float = 0.0
    fps: float = 0.0
    model_size_mb: float = 0.0
    per_class_test_ap: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TestInferenceSample:
    """Record of test split inference predictions paired with ground truth.

    Attributes:
        image_filename: Name of the test image sample.
        predictions: List of detected boxes, class labels, and confidence scores.
        ground_truths: List of ground-truth annotated boxes and class labels.
        is_exact_match: Boolean flag indicating if all objects were correctly identified.
    """

    image_filename: str
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    ground_truths: List[Dict[str, Any]] = field(default_factory=list)
    is_exact_match: bool = False


@dataclass
class ExperimentMetadata:
    """Formal scientific metadata and hypothesis formulation for an experiment.

    Attributes:
        exp_id: Canonical experiment identifier (e.g. 'exp1_architecture_battle').
        title: Descriptive publication title.
        hypothesis: Clear formal hypothesis being tested.
        independent_variables: Parameters varied across arms.
        control_variables: Parameters held invariant across all arms.
        evaluation_protocol: Primary metrics determining champion arm.
    """

    exp_id: str
    title: str
    hypothesis: str
    independent_variables: List[str]
    control_variables: List[str]
    evaluation_protocol: str
