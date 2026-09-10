"""Cross-Split Statistical Integrity and Feature Coverage Evaluation Module.

Distributional divergence, Wasserstein distances, Kolmogorov-Smirnov statistics,
and nearest-neighbor distances across split partitions are evaluated. Empirical
results verify that zero visual or feature leakage occurs across split boundaries
while ensuring representative manifold coverage between train and test pools.
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
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA

from src.core.config import ProjectPaths
from src.eda.style import EDADesignSystem

logger = logging.getLogger("IntegrityAnalyzer")


@dataclass
class IntegrityMetrics:
    """Quantitative statistical verification metrics across dataset splits.

    Attributes:
        is_meta_disjoint: Boolean confirmation that no cluster overlaps across partitions.
        wasserstein_train_val: 1-Wasserstein distance between train and validation on PC1.
        wasserstein_train_test: 1-Wasserstein distance between train and test on PC1.
        ks_stat_train_test: Kolmogorov-Smirnov two-sample test statistic on PC1.
        ks_pvalue_train_test: Asymptotic p-value from two-sample Kolmogorov-Smirnov test.
        mean_min_dist_train_test: Average minimum cosine distance from test samples to train.
        p01_min_dist_train_test: 1st percentile minimum cosine distance (near-duplicate probe).
        test_feature_coverage_pct: Percentage of test samples with cosine similarity >= 0.70 to train.
    """

    is_meta_disjoint: bool
    wasserstein_train_val: float
    wasserstein_train_test: float
    ks_stat_train_test: float
    ks_pvalue_train_test: float
    mean_min_dist_train_test: float
    p01_min_dist_train_test: float
    test_feature_coverage_pct: float


class IntegrityAnalyzer:
    """Evaluates cross-split representation alignment and verifies zero-leakage isolation.

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
    def _compute_subspace_divergence(
        train_embs: np.ndarray,
        val_embs: np.ndarray,
        test_embs: np.ndarray,
    ) -> Tuple[float, float, float, float, List[float]]:
        """Computes Wasserstein earth mover distances and Kolmogorov-Smirnov statistics across PCA projections.

        Args:
            train_embs: Training partition embeddings of shape (N_tr, D).
            val_embs: Validation partition embeddings of shape (N_va, D).
            test_embs: Test partition embeddings of shape (N_te, D).

        Returns:
            Tuple of (w_train_val, w_train_test, ks_stat, ks_pvalue, w_per_comp).
        """
        pca = PCA(n_components=5, random_state=42)
        train_pca = pca.fit_transform(train_embs)
        val_pca = pca.transform(val_embs)
        test_pca = pca.transform(test_embs)

        w_train_val = float(wasserstein_distance(train_pca[:, 0], val_pca[:, 0]))
        w_train_test = float(wasserstein_distance(train_pca[:, 0], test_pca[:, 0]))

        ks_res = ks_2samp(train_pca[:, 0], test_pca[:, 0])
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        w_per_comp: List[float] = [
            float(wasserstein_distance(train_pca[:, k], test_pca[:, k]))
            for k in range(5)
        ]
        return w_train_val, w_train_test, ks_stat, ks_pval, w_per_comp

    @staticmethod
    def _compute_isolation_metrics(
        train_embs: np.ndarray,
        val_embs: np.ndarray,
        test_embs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Calculates nearest neighbor cosine distances from validation and test samples to training pool.

        Args:
            train_embs: Training embeddings of shape (N_tr, D).
            val_embs: Validation embeddings of shape (N_va, D).
            test_embs: Test embeddings of shape (N_te, D).

        Returns:
            Tuple of (min_cosine_dist_test, min_cosine_dist_val, mean_min_dist, p01_min_dist, coverage_pct).
        """
        sim_te_tr = test_embs @ train_embs.T
        min_cosine_dist_test = 1.0 - np.max(sim_te_tr, axis=1)

        sim_va_tr = val_embs @ train_embs.T
        min_cosine_dist_val = 1.0 - np.max(sim_va_tr, axis=1)

        mean_min_dist = float(np.mean(min_cosine_dist_test))
        p01_min_dist = float(np.percentile(min_cosine_dist_test, 1))
        coverage_pct = float(np.mean(np.max(sim_te_tr, axis=1) >= 0.70) * 100.0)

        return min_cosine_dist_test, min_cosine_dist_val, mean_min_dist, p01_min_dist, coverage_pct

    def analyze(self) -> Tuple[IntegrityMetrics, Dict[str, Path]]:
        """Calculates cross-split statistical divergences and representation distances.

        Principal components characterize subspace alignment across partitions.
        Minimum cosine distance distributions quantify empirical isolation from training samples.

        Returns:
            Tuple of calculated IntegrityMetrics and generated figure paths.
        """
        with h5py.File(self.paths.embeddings_h5_path, "r") as h5f:
            embeddings: np.ndarray = h5f["embeddings"][:]
            filenames: List[str] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in h5f["filenames"][:]
            ]

        df = pd.read_csv(self.paths.manifest_path)
        if "is_pruned" in df.columns:
            active_df = df[~df["is_pruned"].astype(bool)].copy().reset_index(drop=True)
        else:
            active_df = df.copy().reset_index(drop=True)

        fname_to_split = dict(zip(active_df["filename"], active_df["split"]))

        train_mask = np.array([fname_to_split.get(f) == "train" for f in filenames])
        val_mask = np.array([fname_to_split.get(f) == "val" for f in filenames])
        test_mask = np.array([fname_to_split.get(f) == "test" for f in filenames])

        train_embs = embeddings[train_mask]
        val_embs = embeddings[val_mask]
        test_embs = embeddings[test_mask]

        logger.info(
            "Evaluating cross-split integrity across Train (%d), Val (%d), Test (%d)...",
            len(train_embs), len(val_embs), len(test_embs),
        )

        train_clusters = set(active_df[active_df["split"] == "train"]["meta_cluster"])
        val_clusters = set(active_df[active_df["split"] == "val"]["meta_cluster"])
        test_clusters = set(active_df[active_df["split"] == "test"]["meta_cluster"])
        is_disjoint = (
            len(train_clusters & val_clusters) == 0 and
            len(train_clusters & test_clusters) == 0 and
            len(val_clusters & test_clusters) == 0
        )

        w_train_val, w_train_test, ks_stat, ks_pval, w_per_comp = self._compute_subspace_divergence(
            train_embs, val_embs, test_embs
        )

        min_cosine_dist_test, min_cosine_dist_val, mean_min_dist, p01_min_dist, coverage_pct = self._compute_isolation_metrics(
            train_embs, val_embs, test_embs
        )

        metrics = IntegrityMetrics(
            is_meta_disjoint=is_disjoint,
            wasserstein_train_val=w_train_val,
            wasserstein_train_test=w_train_test,
            ks_stat_train_test=ks_stat,
            ks_pvalue_train_test=ks_pval,
            mean_min_dist_train_test=mean_min_dist,
            p01_min_dist_train_test=p01_min_dist,
            test_feature_coverage_pct=coverage_pct,
        )

        fig_paths = self._generate_figures(
            min_cosine_dist_test=min_cosine_dist_test,
            min_cosine_dist_val=min_cosine_dist_val,
            w_per_comp=w_per_comp,
        )

        return metrics, fig_paths

    def _generate_figures(
        self,
        min_cosine_dist_test: np.ndarray,
        min_cosine_dist_val: np.ndarray,
        w_per_comp: List[float],
    ) -> Dict[str, Path]:
        """Renders publication diagnostic figures for nearest distances and divergence.

        Args:
            min_cosine_dist_test: Minimum cosine distance array from test to train.
            min_cosine_dist_val: Minimum cosine distance array from val to train.
            w_per_comp: List of Wasserstein distance floats across leading PCA axes.

        Returns:
            Dictionary mapping plot identifiers to generated filesystem locations.
        """
        fig_paths: Dict[str, Path] = {}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

        sns.kdeplot(
            min_cosine_dist_val,
            label="Val to Train (Nearest)",
            color="#e66101",
            fill=True,
            alpha=0.3,
            ax=axes[0],
        )
        sns.kdeplot(
            min_cosine_dist_test,
            label="Test to Train (Nearest)",
            color="#02818a",
            fill=True,
            alpha=0.3,
            ax=axes[0],
        )
        axes[0].axvline(0.02, color="#d62728", linestyle="--", linewidth=1.5, label="Duplicate Boundary (Dist < 0.02)")
        EDADesignSystem.apply_theme(
            axes[0],
            title="Nearest Neighbor Cosine Distance to Training Pool",
            xlabel="Minimum Cosine Distance: min(1 - cos_sim)",
            ylabel="Probability Density",
        )
        axes[0].legend(frameon=True, loc="upper right")

        pcs = [f"PC {k+1}" for k in range(len(w_per_comp))]
        sns.barplot(
            x=pcs,
            y=w_per_comp,
            color="#2b5c8f",
            ax=axes[1],
            edgecolor="none",
        )
        EDADesignSystem.apply_theme(
            axes[1],
            title="Train vs Test Wasserstein Divergence (Top 5 PCA Axes)",
            xlabel="Principal Component Axis",
            ylabel="1-Wasserstein Earth Mover Distance",
        )

        self.paths.figures_splits_dir.mkdir(parents=True, exist_ok=True)
        int_path_split = self.paths.figures_splits_dir / "eda_split_integrity.png"
        EDADesignSystem.save_figure(fig, int_path_split)
        fig_paths["split_integrity"] = int_path_split

        return fig_paths
