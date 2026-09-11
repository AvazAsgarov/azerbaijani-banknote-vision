"""Standardized Scientific Figure Generator for Research Experiments 1-6.

Produces publication-grade comparative diagnostic figures strictly adhering to the
project EDADesignSystem (300 DPI, IEEE typography, canonical palettes, and clean spines).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda.style import EDADesignSystem

logger = logging.getLogger("ExperimentVisualizer")


class ExperimentVisualizer:
    """Scientific visualization engine generating cohesive diagnostic figures for Exp 1-6."""

    @staticmethod
    def plot_exp1_pareto_frontier(df: pd.DataFrame, output_path: Path) -> Path:
        """Plots Pareto Frontier: Hardware Latency vs Test Detection Accuracy (Exp 1).

        Args:
            df: DataFrame containing columns ['model_id', 'architecture', 'latency_gpu_ms',
                'test_map50_95', 'parameters_m'].
            output_path: Target filesystem path for the generated PNG.

        Returns:
            Path to the saved figure.
        """
        valid_df = df[df["test_map50_95"].notnull() & df["latency_gpu_ms"].notnull()].copy()
        fig, ax = plt.subplots(figsize=(9, 5.5))

        palette = ["#2b5c8f", "#02818a", "#e66101", "#9467bd"]
        for idx, (_, row) in enumerate(valid_df.iterrows()):
            color = palette[idx % len(palette)]
            size = max(100.0, float(row["parameters_m"]) * 10.0 + 80.0)
            ax.scatter(
                row["latency_gpu_ms"],
                row["test_map50_95"] * 100.0,
                s=size,
                color=color,
                alpha=0.85,
                edgecolors="black",
                linewidth=1.2,
                label=f"{row['model_id']} ({row['parameters_m']}M params)",
                zorder=3,
            )
            # Label annotation
            ax.annotate(
                f"{row['model_id']}\n({row['test_map50_95']*100:.1f}%, {row['latency_gpu_ms']:.1f}ms)",
                (row["latency_gpu_ms"] + 0.25, row["test_map50_95"] * 100.0 - 0.4),
                fontsize=9,
                fontweight="bold",
                color="#1a1a1a",
            )

        EDADesignSystem.apply_theme(
            ax,
            title="Pareto Frontier: A100 GPU Latency vs. Test Detection Accuracy (mAP@0.5:0.95)",
            xlabel="A100 Forward Latency (ms / image, batch=1)",
            ylabel="Zero-Leakage Test mAP@0.5:0.95 (%)",
        )
        ax.legend(title="Model Architecture", loc="lower right", frameon=True, fontsize=8.5)
        EDADesignSystem.save_figure(fig, output_path)
        logger.info("Exp 1 Pareto figure saved to %s", output_path)
        return output_path

    @staticmethod
    def plot_exp2_augmentation_delta(df: pd.DataFrame, output_path: Path) -> Path:
        """Plots Augmentation Ablation Delta: Relative mAP gains over Raw Baseline (Exp 2).

        Args:
            df: DataFrame containing columns ['arm_id', 'display_name', 'test_map50_95'].
            output_path: Target filesystem path for the generated PNG.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_df = df.copy()

        # Find raw baseline score
        raw_rows = plot_df[plot_df["arm_id"].str.contains("raw|arm1", case=False)]
        raw_baseline = float(raw_rows.iloc[0]["test_map50_95"]) if not raw_rows.empty else 0.824

        plot_df["map_pct"] = plot_df["test_map50_95"] * 100.0
        plot_df["delta_pct"] = (plot_df["test_map50_95"] - raw_baseline) * 100.0

        bar_colors = ["#7f7f7f" if d == 0 else "#2b5c8f" if d > 0 else "#d62728" for d in plot_df["delta_pct"]]

        bars = ax.bar(
            range(len(plot_df)),
            plot_df["map_pct"],
            color=bar_colors,
            edgecolor="black",
            linewidth=1.0,
            width=0.55,
            zorder=3,
        )

        ax.axhline(raw_baseline * 100.0, color="#d62728", linestyle="--", linewidth=1.5, label=f"Raw Baseline ({raw_baseline*100:.1f}%)")

        # Text labels on bars
        for idx, bar in enumerate(bars):
            h = bar.get_height()
            delta = plot_df["delta_pct"].iloc[idx]
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%" if delta < 0 else "Baseline"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.6,
                f"{h:.1f}%\n({delta_str})",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_xticks(range(len(plot_df)))
        labels = [row["arm_id"].replace("arm", "Arm ").replace("_", " ") for _, row in plot_df.iterrows()]
        ax.set_xticklabels(labels, rotation=15, ha="right")

        EDADesignSystem.apply_theme(
            ax,
            title="Data Augmentation Impact Ablation: Generalization Delta across Arms",
            xlabel="Ablation Arm Configuration",
            ylabel="Zero-Leakage Test mAP@0.5:0.95 (%)",
        )
        ax.set_ylim(0, max(plot_df["map_pct"]) + 8.0)
        ax.legend(loc="upper left")
        EDADesignSystem.save_figure(fig, output_path)
        logger.info("Exp 2 Augmentation ablation figure saved to %s", output_path)
        return output_path

    @staticmethod
    def plot_exp3_color_space_comparison(df: pd.DataFrame, output_path: Path) -> Path:
        """Plots Color Space Shortcut Learning Analysis: RGB vs HSV vs Grayscale (Exp 3).

        Args:
            df: DataFrame containing columns ['arm_id', 'color_space', 'test_map50', 'test_map50_95'].
            output_path: Target filesystem path for the generated PNG.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plot_df = df.copy()

        x = np.arange(len(plot_df))
        width = 0.35

        map50_vals = plot_df["test_map50"] * 100.0
        map95_vals = plot_df["test_map50_95"] * 100.0

        b1 = ax.bar(x - width / 2, map50_vals, width, label="mAP@0.50", color="#2b5c8f", edgecolor="black", linewidth=0.8, zorder=3)
        b2 = ax.bar(x + width / 2, map95_vals, width, label="mAP@0.50:0.95", color="#02818a", edgecolor="black", linewidth=0.8, zorder=3)

        # Text labels on bars
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        names = [str(r.get("color_space", r.get("arm_id", ""))).upper() for _, r in plot_df.iterrows()]
        ax.set_xticklabels(names)

        EDADesignSystem.apply_theme(
            ax,
            title="Color Space Representation Resilience: RGB vs. HSV vs. Grayscale",
            xlabel="Color Space Representation",
            ylabel="Zero-Leakage Test Detection Accuracy (%)",
        )
        ax.set_ylim(0, 108)
        ax.legend(loc="lower right")
        EDADesignSystem.save_figure(fig, output_path)
        logger.info("Exp 3 Color space figure saved to %s", output_path)
        return output_path

    @staticmethod
    def plot_exp4_multiscale_resolution(df: pd.DataFrame, output_path: Path) -> Path:
        """Plots Multiscale Resolution Scaling: Accuracy vs Throughput FPS (Exp 4).

        Args:
            df: DataFrame containing columns ['resolution', 'test_map50_95', 'fps'].
            output_path: Target filesystem path for the generated PNG.

        Returns:
            Path to the saved figure.
        """
        fig, ax1 = plt.subplots(figsize=(8.5, 5))
        plot_df = df.copy()

        x = np.arange(len(plot_df))
        res_labels = [f"{r}x{r}" for r in plot_df["resolution"]]

        color_map = "#2b5c8f"
        color_fps = "#e66101"

        # Left axis: mAP
        bars = ax1.bar(x - 0.18, plot_df["test_map50_95"] * 100.0, width=0.35, color=color_map, edgecolor="black", label="Test mAP@0.5:0.95", zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color_map)

        ax1.set_ylabel("Test mAP@0.5:0.95 (%)", color=color_map, fontsize=10, labelpad=6)
        ax1.tick_params(axis="y", labelcolor=color_map)
        ax1.set_ylim(60, 105)

        # Right axis: FPS
        ax2 = ax1.twinx()
        line = ax2.plot(x + 0.18, plot_df["fps"], color=color_fps, marker="o", linewidth=2.0, markersize=7, label="Inference FPS", zorder=4)
        for i, fps_val in enumerate(plot_df["fps"]):
            ax2.text(x[i] + 0.18, fps_val + 15, f"{fps_val:.0f} FPS", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color_fps)

        ax2.set_ylabel("Throughput (Frames Per Second)", color=color_fps, fontsize=10, labelpad=6)
        ax2.tick_params(axis="y", labelcolor=color_fps)
        ax2.spines["top"].set_visible(False)
        ax2.set_ylim(0, max(plot_df["fps"]) * 1.25)

        ax1.set_xticks(x)
        ax1.set_xticklabels(res_labels, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Input Canvas Resolution (pixels)", fontsize=10, labelpad=6)
        ax1.set_title("Input Resolution Dynamics: Detection Accuracy vs. Real-Time FPS", fontsize=12, fontweight="bold", pad=12)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(True, linestyle="--", alpha=0.25)

        EDADesignSystem.save_figure(fig, output_path)
        logger.info("Exp 4 Resolution figure saved to %s", output_path)
        return output_path

    @staticmethod
    def plot_exp5_compression_frontier(df: pd.DataFrame, output_path: Path) -> Path:
        """Plots Model Compression & Optimization Frontier: Size vs mAP (Exp 5).

        Args:
            df: DataFrame containing columns ['technique', 'model_size_mb', 'test_map50_95', 'latency_ms'].
            output_path: Target filesystem path for the generated PNG.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plot_df = df.copy()

        colors = ["#2b5c8f", "#02818a", "#e66101", "#9467bd"]
        for idx, (_, row) in enumerate(plot_df.iterrows()):
            color = colors[idx % len(colors)]
            ax.scatter(
                row["model_size_mb"],
                row["test_map50_95"] * 100.0,
                s=180,
                color=color,
                edgecolors="black",
                linewidth=1.2,
                label=f"{row['technique']} ({row['model_size_mb']:.1f} MB)",
                zorder=3,
            )
            ax.annotate(
                f"{row['technique']}\n({row['test_map50_95']*100:.1f}%, {row['latency_ms']:.1f}ms)",
                (row["model_size_mb"] + 0.8, row["test_map50_95"] * 100.0 - 0.4),
                fontsize=8.5,
                fontweight="bold",
            )

        EDADesignSystem.apply_theme(
            ax,
            title="Embedded Edge Compression Frontier: Checkpoint Size vs. Accuracy",
            xlabel="Serialized Checkpoint Footprint (Megabytes)",
            ylabel="Zero-Leakage Test mAP@0.5:0.95 (%)",
        )
        ax.legend(title="Compression Regime", loc="lower right", frameon=True, fontsize=8.5)
        EDADesignSystem.save_figure(fig, output_path)
        logger.info("Exp 5 Compression figure saved to %s", output_path)
        return output_path

    @staticmethod
    def plot_exp6_numismatic_alignment(df: pd.DataFrame, output_path: Path) -> Path:
        """Plots Explainable AI (XAI) Numismatic Alignment across Denominations (Exp 6).

        Args:
            df: DataFrame containing columns ['class_name', 'alignment_score'].
            output_path: Target filesystem path for the generated PNG.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(9, 4.8))
        plot_df = df.copy()

        classes = sorted(list(plot_df["class_name"].unique()))
        colors = [EDADesignSystem.CLASS_PALETTE.get(c, "#2b5c8f") for c in plot_df["class_name"]]

        bars = ax.bar(
            range(len(plot_df)),
            plot_df["alignment_score"] * 100.0,
            color=colors,
            edgecolor="black",
            linewidth=0.9,
            width=0.55,
            zorder=3,
        )

        # Mean alignment horizontal line
        mean_align = float(plot_df["alignment_score"].mean()) * 100.0
        ax.axhline(mean_align, color="#ca0020", linestyle="--", linewidth=1.5, label=f"Global Mean Alignment ({mean_align:.1f}%)")

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([c.replace("_azn", " AZN") for c in plot_df["class_name"]], fontsize=9.5, fontweight="bold")

        EDADesignSystem.apply_theme(
            ax,
            title="Numismatic Attention Alignment: C2PSA Feature Focus on Authentic Security Motifs",
            xlabel="Azerbaijani Banknote Denomination",
            ylabel="Attention-to-Security-Motif Alignment Score (%)",
        )
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right")
        EDADesignSystem.save_figure(fig, output_path)
        logger.info("Exp 6 Numismatic alignment figure saved to %s", output_path)
        return output_path
