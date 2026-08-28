"""Publication-Grade Scientific Design and Typography System.

Standardized color palettes, categorical denomination mappings, and publication
typography parameters ensure cohesive visual aesthetics across all exported figures.
Export utilities enforce 300 DPI resolution, tight bounding boxes, and clean spines.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt


class EDADesignSystem:
    """Scientific visualization design system for exploratory and diagnostic plotting.

    Attributes:
        CLASS_PALETTE: Canonical categorical colors assigned to banknote denominations.
        SPLIT_PALETTE: Consistent partition colors distinguishing train, val, and test.
        DIVERGING_PALETTE: Diverging colormap name for zero-centered correlations.
        SEQUENTIAL_PALETTE: Sequential colormap name for density heatmaps.
    """

    # Categorical palette assigned to banknote denomination classes
    CLASS_PALETTE: Dict[str, str] = {
        "001_azn": "#1f77b4",
        "005_azn": "#ff7f0e",
        "010_azn": "#2ca02c",
        "020_azn": "#d62728",
        "050_azn": "#9467bd",
        "100_azn": "#8c564b",
        "200_azn": "#e377c2",
    }

    # Partition palette assigned to dataset splits
    SPLIT_PALETTE: Dict[str, str] = {
        "train": "#2b5c8f",
        "val": "#e66101",
        "test": "#02818a",
        "Train": "#2b5c8f",
        "Val": "#e66101",
        "Test": "#02818a",
    }

    # Sequential divergence palette for density and correlation heatmaps
    DIVERGING_PALETTE: str = "vlag"
    SEQUENTIAL_PALETTE: str = "mako"

    @classmethod
    def apply_theme(
        cls,
        ax: plt.Axes,
        title: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        enable_grid: bool = True,
    ) -> None:
        """
        Standardized styling rules are applied to a Matplotlib axes instance.

        Args:
            ax: Target Matplotlib axes object.
            title: Formatted figure title.
            xlabel: Optional horizontal axis label.
            ylabel: Optional vertical axis label.
            enable_grid: Flag controlling dashed grid visibility.
        """
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=6)

        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if enable_grid:
            ax.grid(True, linestyle="--", alpha=0.25, zorder=0)

    @classmethod
    def save_figure(cls, fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
        """
        High-resolution figure rendering is serialized to disk.

        Args:
            fig: Matplotlib Figure instance.
            output_path: Target filesystem path for the generated image.
            dpi: Dots per inch resolution parameter.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
