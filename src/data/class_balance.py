"""Class balance analyser and inverse-frequency sampler weights for AZN dataset."""

from pathlib import Path
from collections import Counter

NUM_CLASSES = 7
CLASS_NAMES = ["1 AZN", "5 AZN", "10 AZN", "20 AZN", "50 AZN", "100 AZN", "200 AZN"]


def count_instances(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for txt in sorted(labels_dir.rglob("*.txt")):
        for line in txt.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1
    return counts


def compute_weights(counts: Counter, num_classes: int = NUM_CLASSES) -> list:
    total = sum(counts.values()) or 1
    return [total / (num_classes * counts.get(i, 0)) if counts.get(i, 0) > 0 else 0.0 for i in range(num_classes)]


def imbalance_ratio(counts: Counter) -> float:
    vals = [v for v in counts.values() if v > 0]
    return max(vals) / min(vals) if len(vals) >= 2 else 1.0
