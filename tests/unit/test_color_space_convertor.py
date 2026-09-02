"""Unit tests for colour space convertor."""
import pytest
from src.eda.color_space_convertor import rgb_to_hsv, rgb_to_grayscale, hsv_to_rgb

class TestRGBtoHSV:
    def test_pure_red(self):
        h, s, v = rgb_to_hsv(1.0, 0.0, 0.0)
        assert h == pytest.approx(0.0, abs=1e-5)
        assert s == pytest.approx(1.0, abs=1e-5)
    def test_white(self):
        _, s, v = rgb_to_hsv(1.0, 1.0, 1.0)
        assert s == pytest.approx(0.0, abs=1e-5)

class TestGrayscale:
    def test_rec601_red(self):
        assert rgb_to_grayscale(1.0, 0.0, 0.0) == pytest.approx(0.299, abs=1e-5)

class TestRoundtrip:
    @pytest.mark.parametrize("rgb", [(0.8, 0.2, 0.4), (0.1, 0.9, 0.3)])
    def test_hsv_roundtrip(self, rgb):
        r2, g2, b2 = hsv_to_rgb(*rgb_to_hsv(*rgb))
        assert r2 == pytest.approx(rgb[0], abs=1e-5)
        assert g2 == pytest.approx(rgb[1], abs=1e-5)
        assert b2 == pytest.approx(rgb[2], abs=1e-5)
