"""Unit tests for EigenCAM attention diagnostic configuration."""
import pytest
from src.xai.eigencam_config import EigenCAMConfig

class TestEigenCAMConfig:
    def test_two_target_layers(self):
        assert len(EigenCAMConfig().target_layers) == 2
    def test_c2psa_layer_present(self):
        assert any("22" in l for l in EigenCAMConfig().target_layers)
    def test_alignment_above_threshold(self):
        cfg = EigenCAMConfig()
        assert cfg.empirical_alignment_score >= cfg.alignment_threshold
    def test_alignment_plus_background_is_one(self):
        cfg = EigenCAMConfig()
        assert cfg.empirical_alignment_score + cfg.empirical_background_score == pytest.approx(1.0, abs=1e-4)
    def test_175_samples(self):
        assert EigenCAMConfig().num_samples == 175
