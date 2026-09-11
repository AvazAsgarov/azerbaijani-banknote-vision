"""
Unit tests for asymmetric banknote loss function.
Verifies cost-sensitive weighting, bounding box regression, and objectness balancing.
"""

import pytest
import torch

from src.tinyml.loss import AsymmetricBanknoteLoss


@pytest.mark.unit
class TestAsymmetricBanknoteLoss:
    """Test suite verifying asymmetric loss mechanics and cost penalties."""

    def test_loss_initialization_penalty_coefficients(self):
        """Verify penalty vector registers escalating weights across currency denominations.

        Args:
            None.

        Returns:
            None.
        """
        criterion = AsymmetricBanknoteLoss(num_classes=7)

        assert criterion.beta_c.shape == torch.Size([7])
        assert criterion.beta_c[0].item() == 1.0
        assert criterion.beta_c[1].item() == 1.0
        assert criterion.beta_c[2].item() == 2.0
        assert criterion.beta_c[3].item() == 2.0
        assert criterion.beta_c[4].item() == 4.0
        assert criterion.beta_c[5].item() == 6.0
        assert criterion.beta_c[6].item() == 6.0

    def test_loss_computation_zero_target_background(self):
        """Verify loss computation handles pure background batches stably without NaN errors.

        Args:
            None.

        Returns:
            None.
        """
        criterion = AsymmetricBanknoteLoss(num_classes=7)
        pred_s16 = torch.randn(2, 3, 10, 10, 12)
        pred_s32 = torch.randn(2, 3, 5, 5, 12)
        target_s16 = torch.zeros(2, 3, 10, 10, 12)
        target_s32 = torch.zeros(2, 3, 5, 5, 12)

        loss, metrics = criterion(pred_s16, pred_s32, target_s16, target_s32)

        assert torch.isfinite(loss)
        assert loss.item() > 0.0
        assert metrics["loss_box"] == 0.0
        assert metrics["loss_cls"] == 0.0
        assert metrics["loss_obj"] > 0.0

    def test_loss_computation_synthetic_positive_targets(self):
        """Verify loss tracks localization, objectness, and classification on positive targets.

        Args:
            None.

        Returns:
            None.
        """
        criterion = AsymmetricBanknoteLoss(num_classes=7)
        pred_s16 = torch.randn(2, 3, 10, 10, 12)
        pred_s32 = torch.randn(2, 3, 5, 5, 12)
        target_s16 = torch.zeros(2, 3, 10, 10, 12)
        target_s32 = torch.zeros(2, 3, 5, 5, 12)

        target_s16[0, 1, 4, 4, 0] = 0.5
        target_s16[0, 1, 4, 4, 1] = 0.5
        target_s16[0, 1, 4, 4, 2] = 0.1
        target_s16[0, 1, 4, 4, 3] = -0.1
        target_s16[0, 1, 4, 4, 4] = 1.0
        target_s16[0, 1, 4, 4, 5 + 6] = 1.0

        loss, metrics = criterion(pred_s16, pred_s32, target_s16, target_s32)

        assert torch.isfinite(loss)
        assert metrics["loss_box"] > 0.0
        assert metrics["loss_cls"] > 0.0
        assert metrics["loss_obj"] > 0.0

    def test_asymmetric_penalty_escalation_on_high_denominations(self):
        """Verify high-denomination false positives generate significantly higher loss penalties.

        Args:
            None.

        Returns:
            None.
        """
        criterion = AsymmetricBanknoteLoss(num_classes=7)
        target = torch.zeros(1, 1, 1, 1, 12)
        target[0, 0, 0, 0, 4] = 1.0
        target[0, 0, 0, 0, 5 + 0] = 1.0

        pred_low = torch.zeros(1, 1, 1, 1, 12)
        pred_low[0, 0, 0, 0, 5 + 1] = 3.0

        pred_high = torch.zeros(1, 1, 1, 1, 12)
        pred_high[0, 0, 0, 0, 5 + 6] = 3.0

        loss_low, _, _, cls_low = criterion._compute_head_loss(pred_low, target)
        loss_high, _, _, cls_high = criterion._compute_head_loss(pred_high, target)

        assert cls_high.item() > cls_low.item()
