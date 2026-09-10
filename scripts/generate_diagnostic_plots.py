"""
Generates publication-grade diagnostic figures from extracted DINOv2 embeddings.

Adheres strictly to Matplotlib's Object-Oriented API and cohesive Seaborn aesthetics.
Outputs:
1. 'dinov2_extraction_telemetry.png' - L2-norm distribution and feature variance profile.
2. 'dinov2_sample_similarity_heatmap.png' - 50-sample pairwise cosine similarity heatmap.
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths
from src.eda.style import EDADesignSystem


def main() -> None:
    paths = ProjectPaths()
    h5_path = paths.embeddings_h5_path

    with h5py.File(h5_path, "r") as h5f:
        embeddings = h5f["embeddings"][:]
        filenames = [f.decode("utf-8") if isinstance(f, bytes) else str(f) for f in h5f["filenames"][:]]
        class_names = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in h5f["class_names"][:]]

    print(f"Loaded embeddings: {embeddings.shape} from {h5_path}")

    # 1. Telemetry Figure (L2 Norms & Coordinate Variances)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Plot 1: L2 Norm Verification (normalized to 1.0)
    ax1 = axes[0]
    l2_norms = np.linalg.norm(embeddings, axis=1)
    sns.histplot(l2_norms, bins=30, color="#2b5c8f", kde=True, ax=ax1)
    EDADesignSystem.apply_theme(
        ax1,
        title=f"Embedding $L_2$-Norm Distribution ($N={len(embeddings)}$)",
        xlabel="Vector $L_2$-Norm ($\\|z_i\\|_2$)",
        ylabel="Sample Frequency",
    )
    ax1.axvline(1.0, color="#ca0020", ls="--", lw=1.5, label="Unit Hypersphere ($1.0$)")
    ax1.legend(loc="upper right")

    # Plot 2: 1024-d Coordinate Variance Distribution
    ax2 = axes[1]
    coord_stds = np.std(embeddings, axis=0)
    sns.histplot(coord_stds, bins=35, color="#02818a", kde=True, ax=ax2)
    mean_std = float(np.mean(coord_stds))
    EDADesignSystem.apply_theme(
        ax2,
        title="DINOv2 ViT-L/14 Feature Dimension Std Deviation",
        xlabel="Standard Deviation across Samples",
        ylabel="Dimension Count ($d=1024$)",
    )
    ax2.axvline(mean_std, color="#e66101", ls="--", lw=1.5, label=f"Mean Std: {mean_std:.4f}")
    ax2.legend(loc="upper right")

    out1_sub = paths.figures_embeddings_dir / "dinov2_extraction_telemetry.png"
    EDADesignSystem.save_figure(fig, out1_sub)
    print(f"Saved '{out1_sub}'.")

    # 2. Pairwise Cosine Similarity Heatmap (50 Sample Matrix)
    np.random.seed(42)
    sample_indices = np.random.choice(len(embeddings), size=50, replace=False)
    sample_embs = embeddings[sample_indices]
    sample_sim = sample_embs @ sample_embs.T

    fig_sim, ax_sim = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        sample_sim,
        cmap="crest",
        vmin=0.2,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Cosine Affinity $\\cos(z_i, z_j)$", "shrink": 0.8},
        ax=ax_sim,
    )
    EDADesignSystem.apply_theme(
        ax_sim,
        title="Pairwise Cosine Affinity (50 Stratified Samples)",
        xlabel="Sample Index",
        ylabel="Sample Index",
        enable_grid=False,
    )
    out2_sub = paths.figures_embeddings_dir / "dinov2_sample_similarity_heatmap.png"
    EDADesignSystem.save_figure(fig_sim, out2_sub)
    print(f"Saved '{out2_sub}'.")


if __name__ == "__main__":
    main()
