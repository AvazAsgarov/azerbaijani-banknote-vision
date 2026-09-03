"""Unit tests for the augmentation policy registry."""

import pytest
from src.data.augment_policy import get_policy, POLICIES


class TestAugmentPolicy:
    def test_all_canonical_policies_retrievable(self):
        for name in ("raw", "geometric", "photometric", "composite"):
            p = get_policy(name)
            assert isinstance(p, dict)

    def test_raw_policy_has_zero_augmentation(self):
        p = get_policy("raw")
        assert p["fliplr"] == 0.0
        assert p["mosaic"] == 0.0

    def test_composite_has_all_augmentations(self):
        p = get_policy("composite")
        assert p["fliplr"] > 0
        assert p["mosaic"] > 0
        assert p["hsv_s"] > 0

    def test_get_policy_returns_copy(self):
        p1 = get_policy("geometric")
        p1["fliplr"] = 99.9
        assert get_policy("geometric")["fliplr"] != 99.9

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            get_policy("nonexistent")
