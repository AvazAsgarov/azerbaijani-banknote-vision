"""Unit tests for the class balance analyser."""

from pathlib import Path
from collections import Counter
import pytest
from src.data.class_balance import count_instances, compute_weights, imbalance_ratio


def _write(tmp, name, lines):
    (tmp / name).write_text("\n".join(lines))


class TestClassBalance:
    def test_count_instances_correct(self, tmp_path):
        _write(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.1", "2 0.3 0.4 0.2 0.1"])
        _write(tmp_path, "b.txt", ["0 0.6 0.6 0.2 0.1"])
        counts = count_instances(tmp_path)
        assert counts[0] == 2
        assert counts[2] == 1

    def test_imbalance_ratio_perfect_balance(self):
        counts = Counter({i: 100 for i in range(7)})
        assert imbalance_ratio(counts) == pytest.approx(1.0)

    def test_imbalance_ratio_extreme(self):
        counts = Counter({0: 1, 6: 100})
        assert imbalance_ratio(counts) == pytest.approx(100.0)

    def test_compute_weights_inverse_frequency(self):
        counts = Counter({0: 100, 1: 50})
        weights = compute_weights(counts, num_classes=2)
        assert weights[0] < weights[1]

    def test_zero_count_class_gets_zero_weight(self):
        counts = Counter({0: 100})
        weights = compute_weights(counts, num_classes=2)
        assert weights[1] == 0.0
