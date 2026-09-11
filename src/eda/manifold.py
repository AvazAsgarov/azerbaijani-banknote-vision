"""Manifold Topology and Latent Representation Analysis Module.

High-dimensional DINOv2 representations undergo geometric analysis through UMAP
and t-SNE projections, silhouette and Davies-Bouldin cluster metrics, pairwise
cosine affinity heatmaps, and Local Outlier Factor anomaly scoring.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor
import umap

from src.core.config import ProjectPaths
from src.eda.style import EDADesignSystem

logger = logging.getLogger("ManifoldAnalyzer")


@dataclass
class ManifoldMetrics:
    """Quantitative evaluation metrics describing latent space structure.

    Attributes:
        silhouette_cosine: Global silhouette coefficient computed under cosine metric.
        davies_bouldin: Davies-Bouldin cluster separation index (lower denotes tighter clusters).
        calinski_harabasz: Variance ratio criterion evaluating dispersion across clusters.
        mean_intra_class_sim: Expected cosine similarity among representations sharing denomination.
        mean_inter_class_sim: Expected cosine similarity among representations differing in denomination.
        class_similarity_matrix: Contingency mapping of pairwise denomination mean affinities.
        top_outlier_filenames: List of filenames exhibiting highest Local Outlier Factor scores.
        top_outlier_scores: Numerical outlier factor scores for detected anomalous samples.
        top_outlier_classes: Class denomination labels corresponding to anomalous samples.
    """

    silhouette_cosine: float
    davies_bouldin: float
    calinski_harabasz: float
    mean_intra_class_sim: float
    mean_inter_class_sim: float
    class_similarity_matrix: Dict[str, Dict[str, float]]
    top_outlier_filenames: List[str]
    top_outlier_scores: List[float]
    top_outlier_classes: List[str]


class ManifoldAnalyzer:
    """Investigates representation geometry and topological cluster quality.

    Attributes:
        paths: Configuration paths resolving dataset files, embeddings, and reports.
    """

    def __init__(self, paths: Optional[ProjectPaths] = None) -> None:
        """Initializes analyzer with standardized filesystem paths.

        Args:
            paths: Optional project configuration paths containing manifests and features.
        """
        self.paths = paths or ProjectPaths()

    @staticmethod
    def _compute_intra_inter_similarity(
        sim_matrix: np.ndarray,
        labels: np.ndarray,
        unique_classes: List[str],
    ) -> Tuple[List[float], List[float], Dict[str, Dict[str, float]]]:
        """Calculates intra-class cohesion and inter-class separation statistics across affinity matrix.

        Args:
            sim_matrix: Pairwise cosine similarity matrix of shape (N, N).
            labels: Integer class indices of length N.
            unique_classes: Sorted denomination class name strings.

        Returns:
            Tuple of (intra-class similarities list, inter-class similarities list, nested class affinity mapping).
        """
        intra_sims: List[float] = []
        inter_sims: List[float] = []
        class_means: Dict[str, Dict[str, float]] = {c: {} for c in unique_classes}

        for i, c1 in enumerate(unique_classes):
            idx1 = np.where(labels == i)[0]
            for j, c2 in enumerate(unique_classes):
                idx2 = np.where(labels == j)[0]
                sub_sim = sim_matrix[np.ix_(idx1, idx2)]
                if i == j:
                    mask = ~np.eye(len(idx1), dtype=bool)
                    if np.any(mask):
                        intra_vals = sub_sim[mask]
                        intra_sims.extend(intra_vals.tolist())
                        class_means[c1][c2] = float(np.mean(intra_vals))
                    else:
                        class_means[c1][c2] = 1.0
                else:
                    inter_sims.extend(sub_sim.flatten().tolist())
                    class_means[c1][c2] = float(np.mean(sub_sim))

        return intra_sims, inter_sims, class_means

    @staticmethod
    def _detect_lof_outliers(
        embeddings: np.ndarray,
        filenames: List[str],
        classes: List[str],
        top_k: int = 10,
    ) -> Tuple[List[str], List[float], List[str]]:
        """Identifies top-k atypical samples via Local Outlier Factor in cosine representation space.

        Args:
            embeddings: Normalized representation matrix of shape (N, D).
            filenames: List of image filenames.
            classes: List of denomination class strings.
            top_k: Number of highest outlier candidates to return.

        Returns:
            Tuple of (outlier filenames list, outlier scores list, outlier class labels list).
        """
        lof = LocalOutlierFactor(n_neighbors=20, metric="cosine")
        lof_scores = -lof.fit_predict(embeddings) * lof.negative_outlier_factor_
        outlier_order = np.argsort(lof_scores)[::-1]
        k = min(top_k, len(filenames))
        top_outliers = [filenames[idx] for idx in outlier_order[:k]]
        top_scores = [float(lof_scores[idx]) for idx in outlier_order[:k]]
        top_classes = [classes[idx] for idx in outlier_order[:k]]
        return top_outliers, top_scores, top_classes

    def analyze(self) -> Tuple[ManifoldMetrics, Dict[str, Path]]:
        """Latent space manifold projections, cluster quality scores, and similarity profiles are computed.

        Dimensionality reduction via UMAP and t-SNE projects feature representations into
        planar coordinate spaces. Pairwise dot products assess intra-class cohesion and
        inter-class separation.

        Returns:
            Tuple of calculated ManifoldMetrics and generated figure paths.
        """
        with h5py.File(self.paths.embeddings_h5_path, "r") as h5f:
            embeddings: np.ndarray = h5f["embeddings"][:]
            filenames: List[str] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in h5f["filenames"][:]
            ]
            classes: List[str] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in h5f["class_names"][:]
            ]

        unique_classes = sorted(list(set(classes)))
        class_to_idx = {c: idx for idx, c in enumerate(unique_classes)}
        labels = np.array([class_to_idx[c] for c in classes])

        logger.info("Manifold projections are computed for %d samples across %d classes.", len(embeddings), len(unique_classes))

        umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        umap_coords = umap_reducer.fit_transform(embeddings)

        tsne_reducer = TSNE(
            n_components=2,
            perplexity=30,
            metric="cosine",
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        tsne_coords = tsne_reducer.fit_transform(embeddings)

        sil_score = float(silhouette_score(embeddings, labels, metric="cosine"))
        db_score = float(davies_bouldin_score(embeddings, labels))
        ch_score = float(calinski_harabasz_score(embeddings, labels))

        sim_matrix = embeddings @ embeddings.T
        intra_sims, inter_sims, class_means = self._compute_intra_inter_similarity(
            sim_matrix, labels, unique_classes
        )

        top_outliers, top_scores, top_classes = self._detect_lof_outliers(
            embeddings, filenames, classes, top_k=10
        )

        metrics = ManifoldMetrics(
            silhouette_cosine=sil_score,
            davies_bouldin=db_score,
            calinski_harabasz=ch_score,
            mean_intra_class_sim=float(np.mean(intra_sims)),
            mean_inter_class_sim=float(np.mean(inter_sims)),
            class_similarity_matrix=class_means,
            top_outlier_filenames=top_outliers,
            top_outlier_scores=top_scores,
            top_outlier_classes=top_classes,
        )

        fig_paths = self._generate_figures(
            umap_coords=umap_coords,
            tsne_coords=tsne_coords,
            classes=classes,
            unique_classes=unique_classes,
            class_means=class_means,
            intra_sims=intra_sims,
            inter_sims=inter_sims,
        )

        return metrics, fig_paths

    def _generate_figures(
        self,
        umap_coords: np.ndarray,
        tsne_coords: np.ndarray,
        classes: List[str],
        unique_classes: List[str],
        class_means: Dict[str, Dict[str, float]],
        intra_sims: List[float],
        inter_sims: List[float],
    ) -> Dict[str, Path]:
        """Renders manifold scatter projections, cross-denomination affinity heatmaps, and similarity distributions.

        Args:
            umap_coords: 2D coordinate array from UMAP projection.
            tsne_coords: 2D coordinate array from t-SNE projection.
            classes: Denomination class labels aligned with samples.
            unique_classes: Sorted list of distinct denomination names.
            class_means: Nested mapping of pairwise mean cosine similarities.
            intra_sims: List of intra-class pairwise cosine similarity values.
            inter_sims: List of inter-class pairwise cosine similarity values.

        Returns:
            Dictionary mapping plot identifiers to generated filesystem locations.
        """
        fig_paths: Dict[str, Path] = {}
        self.paths.figures_embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        df_proj = pd.DataFrame({
            "UMAP_1": umap_coords[:, 0],
            "UMAP_2": umap_coords[:, 1],
            "tSNE_1": tsne_coords[:, 0],
            "tSNE_2": tsne_coords[:, 1],
            "Class": classes,
        })

        sns.scatterplot(
            data=df_proj,
            x="UMAP_1",
            y="UMAP_2",
            hue="Class",
            palette=EDADesignSystem.CLASS_PALETTE,
            alpha=0.65,
            s=22,
            ax=axes[0],
            edgecolor="none",
        )
        EDADesignSystem.apply_theme(
            axes[0],
            title="DINOv2 ViT-L/14 Manifold Geometry (UMAP Projection)",
            xlabel="UMAP Dimension 1",
            ylabel="UMAP Dimension 2",
        )
        axes[0].legend(title="Denomination", loc="upper right", frameon=True, fontsize=8)

        sns.scatterplot(
            data=df_proj,
            x="tSNE_1",
            y="tSNE_2",
            hue="Class",
            palette=EDADesignSystem.CLASS_PALETTE,
            alpha=0.65,
            s=22,
            ax=axes[1],
            edgecolor="none",
        )
        EDADesignSystem.apply_theme(
            axes[1],
            title="DINOv2 ViT-L/14 Latent Clustering (t-SNE Projection)",
            xlabel="t-SNE Dimension 1",
            ylabel="t-SNE Dimension 2",
        )
        axes[1].legend(title="Denomination", loc="upper right", frameon=True, fontsize=8)

        proj_path_emb = self.paths.figures_embeddings_dir / "eda_manifold_projections.png"
        EDADesignSystem.save_figure(fig, proj_path_emb)
        fig_paths["manifold_projections"] = proj_path_emb

        fig, ax = plt.subplots(figsize=(8, 6.5))
        matrix_data = np.array([[class_means[c1][c2] for c2 in unique_classes] for c1 in unique_classes])
        sns.heatmap(
            matrix_data,
            annot=True,
            fmt=".3f",
            cmap=EDADesignSystem.SEQUENTIAL_PALETTE,
            xticklabels=unique_classes,
            yticklabels=unique_classes,
            cbar_kws={"label": "Mean Cosine Similarity"},
            ax=ax,
            linewidths=0.5,
        )
        EDADesignSystem.apply_theme(
            ax,
            title="Cross-Denomination Latent Affinity Matrix (DINOv2)",
            xlabel="Target Denomination",
            ylabel="Reference Denomination",
            enable_grid=False,
        )
        heat_path_emb = self.paths.figures_embeddings_dir / "eda_class_similarity_matrix.png"
        EDADesignSystem.save_figure(fig, heat_path_emb)
        fig_paths["similarity_matrix"] = heat_path_emb

        fig, ax = plt.subplots(figsize=(9, 4.8))
        sample_size = min(len(inter_sims), 10000)
        sub_inter = np.random.choice(inter_sims, size=sample_size, replace=False)
        sns.kdeplot(intra_sims, label=f"Intra-Class (Mean: {np.mean(intra_sims):.3f})", color="#2b5c8f", fill=True, alpha=0.35, ax=ax)
        sns.kdeplot(sub_inter, label=f"Inter-Class (Mean: {np.mean(sub_inter):.3f})", color="#d62728", fill=True, alpha=0.35, ax=ax)
        EDADesignSystem.apply_theme(
            ax,
            title="Latent Separation Profile: Intra-Class vs Inter-Class Cosine Affinity",
            xlabel="Cosine Similarity",
            ylabel="Empirical Probability Density",
        )
        ax.legend(frameon=True, loc="upper left")
        dist_path_emb = self.paths.figures_embeddings_dir / "eda_similarity_distributions.png"
        EDADesignSystem.save_figure(fig, dist_path_emb)
        fig_paths["similarity_distributions"] = dist_path_emb

        return fig_paths
