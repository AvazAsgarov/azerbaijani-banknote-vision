"""Unit tests for model compression utility functions."""
import pytest
from src.eda.compression_utils import compute_compression_ratio, estimate_model_size_mb, bytes_per_element, EXPERIMENT_5_RESULTS

class TestCompressionUtils:
    def test_fp32_ratio_is_one(self):
        assert compute_compression_ratio(80.0, 80.0) == pytest.approx(1.0)
    def test_fp16_ratio_is_two(self):
        assert compute_compression_ratio(40.0, 80.0) == pytest.approx(2.0)
    def test_bytes_fp32(self):
        assert bytes_per_element("FP32") == 4.0
    def test_bytes_int8(self):
        assert bytes_per_element("INT8") == 1.0
    def test_exp5_sizes_decreasing(self):
        sizes = [p.model_size_mb for p in EXPERIMENT_5_RESULTS]
        assert sizes == sorted(sizes, reverse=True)
