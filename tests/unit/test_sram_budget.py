"""Unit tests for SRAM budget calculator."""
import pytest
from src.tinyml.sram_budget import SRAMProfile, check_budget, YOLO_FASTESTV2_PROFILE

class TestSRAMBudget:
    def test_yolo_fits(self):
        assert YOLO_FASTESTV2_PROFILE.fits is True
    def test_headroom_positive(self):
        assert check_budget(YOLO_FASTESTV2_PROFILE)["headroom_kb"] > 0
    def test_oversized_does_not_fit(self):
        assert SRAMProfile(model_weights_kb=400.0, runtime_buffers_kb=64.0, stack_heap_kb=20.0, total_kb=484.0).fits is False
    def test_total_is_sum(self):
        p = YOLO_FASTESTV2_PROFILE
        assert p.total_kb == pytest.approx(p.model_weights_kb + p.runtime_buffers_kb + p.stack_heap_kb, abs=0.1)
