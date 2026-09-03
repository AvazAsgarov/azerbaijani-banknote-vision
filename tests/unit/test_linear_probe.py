"""Unit tests for linear probe configuration and DINOv2 result."""
import pytest
from src.embeddings.linear_probe import LinearProbeConfig, DINOV2_LINEAR_PROBE_RESULT

class TestLinearProbeConfig:
    def test_embedding_dim(self):
        assert LinearProbeConfig().embedding_dim == 1024
    def test_num_classes(self):
        assert LinearProbeConfig().num_classes == 7
    def test_param_count(self):
        cfg = LinearProbeConfig()
        assert cfg.embedding_dim * cfg.num_classes + cfg.num_classes == 7175

class TestLinearProbeResult:
    def test_100_percent_accuracy(self):
        assert DINOV2_LINEAR_PROBE_RESULT.test_accuracy == pytest.approx(1.0, abs=1e-6)
    def test_num_params(self):
        assert DINOV2_LINEAR_PROBE_RESULT.num_params == 7175
