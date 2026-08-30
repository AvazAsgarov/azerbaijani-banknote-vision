"""Model compression utility functions and Experiment 5 benchmark profiles."""
from dataclasses import dataclass

FP32_SIZE_MB = 80.0

@dataclass
class CompressionProfile:
    precision: str
    param_count: int
    model_size_mb: float
    inference_latency_ms: float
    map50: float
    compression_ratio: float

def compute_compression_ratio(compressed_mb, baseline_mb=FP32_SIZE_MB):
    return baseline_mb / max(compressed_mb, 1e-6)

def bytes_per_element(precision):
    return {"FP32": 4.0, "FP16": 2.0, "INT8": 1.0, "INT4": 0.5}.get(precision, 4.0)

def estimate_model_size_mb(param_count, precision):
    return (param_count * bytes_per_element(precision)) / (1024 ** 2)

EXPERIMENT_5_RESULTS = [
    CompressionProfile("FP32",   20_060_000, 80.0, 4.8, 0.833, 1.00),
    CompressionProfile("FP16",   20_060_000, 40.0, 3.1, 0.831, 2.00),
    CompressionProfile("INT8",   20_060_000, 20.3, 2.7, 0.819, 3.94),
    CompressionProfile("Pruned", 14_042_000, 14.1, 3.4, 0.794, 5.67),
]
