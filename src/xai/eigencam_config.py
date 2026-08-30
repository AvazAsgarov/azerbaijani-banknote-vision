"""EigenCAM attention diagnostic configuration for AZN-Vision Experiment 6."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class EigenCAMConfig:
    target_layers: List[str] = field(default_factory=lambda: ["model.model.22", "model.model.9"])
    num_samples: int = 175
    split: str = "test"
    input_size: int = 640
    alignment_threshold: float = 0.50
    empirical_alignment_score: float = 0.5798
    empirical_background_score: float = 0.4202
