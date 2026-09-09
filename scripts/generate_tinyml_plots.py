"""Diagnostic Plot Generator for TinyML ESP32-S3 Edge Deployment.

Generates publication-quality visualizations for TinyML architectures:
1. Speed vs Accuracy Pareto Frontier on ESP32-S3 microcontroller.
2. Multi-Tier Safety Guardrail Architecture pipeline flow.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TinyMLPlots")


def generate_pareto_chart(output_path: Optional[Path] = None) -> Path:
    """Plots framerate vs detection accuracy Pareto Frontier across candidate edge architectures.

    Args:
        output_path: Optional target path for the saved image. Defaults to ProjectPaths reports.

    Returns:
        Path pointing to the written figure file.
    """
    paths = ProjectPaths()
    target_path = output_path or (paths.root_dir / "reports" / "figures" / "tinyml" / "tinyml_esp32s3_pareto.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Candidate models: (Name, FPS, Accuracy/mAP, SRAM_Fit, Marker)
    models = [
        {"name": "FOMO (MNv1 0.1x, 96px)", "fps": 45.5, "acc": 62.0, "sram": True, "color": "#7f7f7f", "marker": "^"},
        {"name": "FOMO (MNv1 0.35x, 96px)", "fps": 23.8, "acc": 71.5, "sram": True, "color": "#17becf", "marker": "^"},
        {"name": "PP-LCNet 0.35x (128px Cls)", "fps": 13.9, "acc": 76.2, "sram": True, "color": "#bcbd22", "marker": "s"},
        {"name": "YOLO-FastestV2 (Tailored SRAM)", "fps": 5.3, "acc": 78.4, "sram": True, "color": "#2ca02c", "marker": "*"},
        {"name": "YOLO-FastestV2 (PSRAM Thrash)", "fps": 3.4, "acc": 78.4, "sram": False, "color": "#d62728", "marker": "o"},
        {"name": "NanoDet-m (ShuffleNetV2 PSRAM)", "fps": 2.3, "acc": 75.8, "sram": False, "color": "#ff7f0e", "marker": "o"},
        {"name": "NanoDet-Plus (192px PSRAM)", "fps": 1.4, "acc": 76.5, "sram": False, "color": "#9467bd", "marker": "o"},
        {"name": "MNv2-SSDLite (192px PSRAM)", "fps": 1.0, "acc": 74.1, "sram": False, "color": "#8c564b", "marker": "o"}
    ]

    for m in models:
        size = 180 if m["marker"] == "*" else (120 if m["marker"] in ["s", "^"] else 100)
        ax.scatter(m["fps"], m["acc"], s=size, c=m["color"], marker=m["marker"], label=m["name"], zorder=5, edgecolors="black", linewidth=1.2)
        offset_x = 0.8 if m["fps"] < 20 else -8.0
        offset_y = 1.0 if m["sram"] else -1.5
        ax.annotate(
            m["name"],
            (m["fps"], m["acc"]),
            xytext=(m["fps"] + offset_x, m["acc"] + offset_y),
            fontsize=8.5,
            fontweight="bold" if m["sram"] and "YOLO" in m["name"] else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=m["color"], alpha=0.85)
        )

    # Draw minimum interactive framerate line (5 FPS)
    ax.axvline(x=5.0, color="#d62728", linestyle="--", linewidth=1.5, label="Interactive Target (5.0 FPS)")

    # Draw Internal SRAM vs PSRAM boundary zone
    ax.axvspan(5.0, 50.0, alpha=0.08, color="green", label="Single-Cycle Internal SRAM Zone")
    ax.axvspan(0.5, 5.0, alpha=0.08, color="red", label="PSRAM Cache Thrashing Zone (< 5 FPS)")

    ax.set_xlabel("ESP32-S3 Execution Framerate (FPS @ 240 MHz)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Detection mAP50 / Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("TinyML Pareto Frontier: Speed vs Accuracy on ESP32-S3 Sense", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0.5, 50.0)
    ax.set_ylim(55.0, 85.0)
    ax.set_xscale("log")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()
    logger.info("Saved Pareto Frontier chart to: %s", target_path)
    return target_path


def generate_safety_guard_chart(output_path: Optional[Path] = None) -> Path:
    """Plots multi-tier safety guardrail validation architecture.

    Args:
        output_path: Optional target path for the saved image. Defaults to ProjectPaths reports.

    Returns:
        Path pointing to the written figure file.
    """
    paths = ProjectPaths()
    target_path = output_path or (paths.root_dir / "reports" / "figures" / "tinyml" / "tinyml_safety_guardrail_flow.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    tiers = [
        "Raw YOLO Detections\n(Candidate Boxes)",
        "Tier 1: Geometry Gate\n(Aspect ratio: 1.4-2.5\nArea >= 8% frame)",
        "Tier 2: Dynamic Gate\n(001 AZN: 0.50\n100 AZN: 0.75\n200 AZN: 0.80)",
        "Tier 3: Temporal Ring\n(3 of 5 match\nSliding Window)",
        "Tier 4: BLE Audio\n(Confirmed Speech\nor Hold Guidance)",
    ]

    x_positions = np.linspace(1, 9, len(tiers))
    y_position = 2.5

    for i, (x, title) in enumerate(zip(x_positions, tiers)):
        box_color = "#2ca02c" if i == 4 else ("#1f77b4" if i == 0 else "#ff7f0e")
        ax.text(
            x,
            y_position,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=box_color, linewidth=2.0),
        )
        if i < len(tiers) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.75, y_position),
                xytext=(x + 0.75, y_position),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=2.0),
            )

    ax.set_xlim(0, 10)
    ax.set_ylim(1, 4)
    ax.axis("off")
    ax.set_title("Multi-Tier Assistive Safety Guardrail Verification Pipeline", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()
    logger.info("Saved Safety Guardrail chart to: %s", target_path)
    return target_path


def main() -> None:
    """CLI orchestrator for TinyML diagnostic plot generation."""
    parser = argparse.ArgumentParser(description="Generate TinyML diagnostic visualizations")
    parser.parse_args()

    generate_pareto_chart()
    generate_safety_guard_chart()


if __name__ == "__main__":
    main()
