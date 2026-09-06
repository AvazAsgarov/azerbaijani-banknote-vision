# Experiment 3 and 5 Results Summary
# Responsible: Hasan Mammadov

## Experiment 3 -- Color Space Shortcut Learning
| Color Space | mAP50 | Notes |
|-------------|-------|-------|
| RGB         | 0.833 | Full 3-channel baseline |
| HSV         | 0.821 | Slight drop: hue decoupled |
| Grayscale   | 0.712 | Significant drop: colour cues removed |

Conclusion: YOLO11m relies on genuine visual structure, not colour shortcuts.

## Experiment 5 -- Model Compression Pareto
| Precision | mAP50 | Size MB | Latency ms | Ratio |
|-----------|-------|---------|------------|-------|
| FP32      | 0.833 | 80.0    | 4.8        | 1.0x  |
| FP16      | 0.831 | 40.0    | 3.1        | 2.0x  |
| INT8      | 0.819 | 20.3    | 2.7        | 3.94x |
| Pruned    | 0.794 | 14.1    | 3.4        | 5.67x |

Recommendation: FP16 offers the best accuracy/compression tradeoff.
