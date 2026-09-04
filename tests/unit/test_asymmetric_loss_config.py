"""Unit tests for asymmetric risk penalty configuration."""
import pytest
from src.core.asymmetric_loss_config import BETA_C, BETA_C_NORMALISED, CLASS_NAMES

class TestAsymmetricLossConfig:
    def test_all_seven_classes(self):
        assert len(BETA_C) == 7
    def test_monotonically_increasing(self):
        p = [BETA_C[i] for i in range(7)]
        for i in range(len(p)-1):
            assert p[i] <= p[i+1]
    def test_200_highest(self):
        assert BETA_C[6] == max(BETA_C.values())
    def test_normalised_mean_is_one(self):
        assert sum(BETA_C_NORMALISED.values()) / len(BETA_C_NORMALISED) == pytest.approx(1.0, abs=1e-6)
    def test_class_names_length(self):
        assert len(CLASS_NAMES) == 7
