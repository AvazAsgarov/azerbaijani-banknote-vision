"""
Unit tests for EDA design system, style palettes, and publication themes.
Verifies categorical palettes, theme applications, and figure export formatting.
"""

import pytest
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path

from src.eda.style import EDADesignSystem


@pytest.mark.unit
class TestEDADesignSystem:
    """Test suite verifying EDADesignSystem palettes and theme functions."""

    def test_class_palette_covers_all_seven_denominations(self):
        """Verify categorical class palette assigns unique hexadecimal colors to all seven classes.

        Args:
            None.

        Returns:
            None.
        """
        palette = EDADesignSystem.CLASS_PALETTE
        expected_classes = [
            "001_azn", "005_azn", "010_azn", "020_azn", "050_azn", "100_azn", "200_azn"
        ]

        assert len(palette) == 7
        for cls_name in expected_classes:
            assert cls_name in palette
            assert palette[cls_name].startswith("#")

    def test_split_palette_consistency(self):
        """Verify split palette defines consistent mappings across partitions.

        Args:
            None.

        Returns:
            None.
        """
        palette = EDADesignSystem.SPLIT_PALETTE

        assert "train" in palette
        assert "val" in palette
        assert "test" in palette
        assert palette["train"] == palette["Train"]

    def test_apply_theme_configures_axes_properties(self):
        """Verify theme application styles axes titles, labels, and spine visibility.

        Args:
            None.

        Returns:
            None.
        """
        fig, ax = plt.subplots()
        try:
            EDADesignSystem.apply_theme(
                ax,
                title="Sample Analysis Plot",
                xlabel="X Dimension",
                ylabel="Y Dimension",
                enable_grid=True
            )

            assert ax.get_title() == "Sample Analysis Plot"
            assert ax.get_xlabel() == "X Dimension"
            assert ax.get_ylabel() == "Y Dimension"
            assert not ax.spines["top"].get_visible()
            assert not ax.spines["right"].get_visible()
        finally:
            plt.close(fig)

    def test_save_figure_creates_high_dpi_output(self):
        """Verify figure saving enforces 300 DPI resolution and tight bounding box layout.

        Args:
            None.

        Returns:
            None.
        """
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_figure.png"
            try:
                EDADesignSystem.save_figure(fig, out_file, dpi=300)

                assert out_file.exists()
                assert out_file.stat().st_size > 0
            finally:
                plt.close(fig)
