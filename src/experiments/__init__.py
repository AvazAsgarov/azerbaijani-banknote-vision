"""Research Experiments Management, Persistence, and Visualization Suite.

Provides unified abstractions for configuring, running, archiving, and visualizing
comparative research experiments 1 through 6 on Azerbaijani Banknote detection.
"""

from src.experiments.persistence import ExperimentArtifactManager
from src.experiments.schema import (
    ArmConfig,
    ExperimentMetadata,
    RunMetrics,
    TestInferenceSample,
)
from src.experiments.visualizer import ExperimentVisualizer

__all__ = [
    "ArmConfig",
    "ExperimentMetadata",
    "RunMetrics",
    "TestInferenceSample",
    "ExperimentArtifactManager",
    "ExperimentVisualizer",
]
