"""Domain-Specific Structural Vulnerability and Deep Dive Analysis Module.

Pairwise denomination confusion vulnerability, physical aspect ratio orientations,
and class imbalance ratios are systematically quantified. Empirical findings guide
hyperparameter bounds including focal loss gamma values, rotation augmentations,
and cost-sensitive loss weighting schemes.
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

from src.core.config import ProjectPaths
from src.eda.style import EDADesignSystem

logger = logging.getLogger("DeepDiveAnalyzer")


@dataclass
class DeepDiveMetrics:
    """Quantitative metrics derived from domain-specific structural investigations.

    Attributes:
        top_confusing_pairs: List of tuples specifying class pairs and separation margins.
        pct_horizontal_banknotes: Percentage of annotated instances having aspect ratio >= 1.25.
        pct_vertical_banknotes: Percentage of annotated instances having aspect ratio <= 0.80.
        pct_square_like: Percentage of instances exhibiting approximately square ratios.
        recommended_focal_gamma: Suggested focusing parameter for hard example focal loss.
        recommended_rotation_degrees: Empirical angular rotation boundary for augmentation.
        class_loss_weights: Balanced inverse frequency weights assigned per denomination.
    """

    top_confusing_pairs: List[Tuple[str, str, float]]
    pct_horizontal_banknotes: float
    pct_vertical_banknotes: float
    pct_square_like: float
    recommended_focal_gamma: float
    recommended_rotation_degrees: float
    class_loss_weights: Dict[str, float]


class DeepDiveAnalyzer:
    """Conducts specialized diagnostic investigations into domain vulnerabilities.

    Attributes:
        paths: Configuration paths resolving dataset directories and embeddings.
    """

    def __init__(self, paths: Optional[ProjectPaths] = None) -> None:
        """Initializes analyzer with standardized filesystem paths.

        Args:
            paths: Optional project configuration paths containing manifests and features.
        """
        self.paths = paths or ProjectPaths()

    @staticmethod
    def _compute_pairwise_margins(
        embeddings: np.ndarray,
        classes: List[str],
        unique_classes: List[str],
    ) -> List[Tuple[str, str, float]]:
        """Calculates pairwise denomination separation margins across latent embeddings.

        Margin is defined as ((intra_1 + intra_2) / 2) - inter. Smaller margins denote
        higher risk of visual confusion.

        Args:
            embeddings: Normalized DINOv2 feature embeddings of shape (N, D).
            classes: Denomination class string labels corresponding to rows of embeddings.
            unique_classes: Sorted distinct list of denomination identifiers.

        Returns:
            Sorted list of tuples (class_1, class_2, separation_margin) ascending by margin.
        """
        sim_matrix = embeddings @ embeddings.T
        classes_arr = np.array(classes)
        pair_margins: List[Tuple[str, str, float]] = []

        for i in range(len(unique_classes)):
            c1 = unique_classes[i]
            idx1 = np.where(classes_arr == c1)[0]
            intra1 = np.mean(sim_matrix[np.ix_(idx1, idx1)])

            for j in range(i + 1, len(unique_classes)):
                c2 = unique_classes[j]
                idx2 = np.where(classes_arr == c2)[0]
                intra2 = np.mean(sim_matrix[np.ix_(idx2, idx2)])
                inter = np.mean(sim_matrix[np.ix_(idx1, idx2)])

                margin = float(((intra1 + intra2) / 2.0) - inter)
                pair_margins.append((c1, c2, margin))

        pair_margins.sort(key=lambda x: x[2])
        return pair_margins

    def _parse_bounding_box_aspect_ratios(
        self,
        active_df: pd.DataFrame,
        unique_classes: List[str],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Extracts bounding box aspect ratios and counts per denomination from YOLO label files.

        Args:
            active_df: Manifest dataframe of active (non-pruned) samples.
            unique_classes: List of canonical denomination class names.

        Returns:
            Tuple of (NumPy array of aspect ratios, dictionary of parsed box counts per class).
        """
        lbl_master_dir = self.paths.master_labels_dir
        aspect_ratios: List[float] = []
        box_counts: Dict[str, int] = {c: 0 for c in unique_classes}

        for _, row in active_df.iterrows():
            fname = str(row["filename"])
            lbl_name = fname.replace(".jpg", ".txt").replace(".png", ".txt")
            lbl_path = lbl_master_dir / lbl_name
            if not lbl_path.exists():
                for s in ["train", "val", "test"]:
                    alt = self.paths.labels_dir / s / lbl_name
                    if alt.exists():
                        lbl_path = alt
                        break

            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            w, h = float(parts[3]), float(parts[4])
                            aspect_ratios.append(max(w, 1e-4) / max(h, 1e-4))
                            if 0 <= cls_id < len(unique_classes):
                                box_counts[unique_classes[cls_id]] += 1

        return np.array(aspect_ratios), box_counts

    def analyze(self) -> Tuple[DeepDiveMetrics, Dict[str, Path]]:
        """Calculates pairwise separation margins, orientation ratios, and class loss weights.

        Pairwise separation margins across denomination representations pinpoint latent
        confusions. Aspect ratios derived from YOLO bounding boxes determine rotation
        augmentation recommendations, while inverse class frequencies determine balanced
        loss weights.

        Returns:
            Tuple containing calculated DeepDiveMetrics and dictionary of figure paths.
        """
        with h5py.File(self.paths.embeddings_h5_path, "r") as h5f:
            embeddings: np.ndarray = h5f["embeddings"][:]
            classes: List[str] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in h5f["class_names"][:]
            ]

        unique_classes = sorted(list(set(classes)))
        pair_margins = self._compute_pairwise_margins(embeddings, classes, unique_classes)
        top_vulnerable = pair_margins[:5]

        df = pd.read_csv(self.paths.manifest_path)
        if "is_pruned" in df.columns:
            active_df = df[~df["is_pruned"].astype(bool)].copy().reset_index(drop=True)
        else:
            active_df = df.copy().reset_index(drop=True)

        ar_arr, box_counts = self._parse_bounding_box_aspect_ratios(active_df, unique_classes)
        pct_horizontal = float(np.mean(ar_arr >= 1.25) * 100.0) if len(ar_arr) > 0 else 0.0
        pct_vertical = float(np.mean(ar_arr <= 0.80) * 100.0) if len(ar_arr) > 0 else 0.0
        pct_square = float(np.mean((ar_arr > 0.80) & (ar_arr < 1.25)) * 100.0) if len(ar_arr) > 0 else 0.0

        class_counts: Dict[str, int] = active_df["class_name"].value_counts().to_dict()
        total_boxes = sum(box_counts.values()) if sum(box_counts.values()) > 0 else sum(class_counts.values())
        num_classes = len(unique_classes)
        loss_weights: Dict[str, float] = {
            c: float(total_boxes / (num_classes * max(1, box_counts.get(c, 1))))
            for c in unique_classes
        }

        rec_gamma = 2.0 if top_vulnerable and top_vulnerable[0][2] < 0.25 else 1.5
        rec_rot = 10.0 if pct_vertical > 10.0 else 5.0

        metrics = DeepDiveMetrics(
            top_confusing_pairs=top_vulnerable,
            pct_horizontal_banknotes=pct_horizontal,
            pct_vertical_banknotes=pct_vertical,
            pct_square_like=pct_square,
            recommended_focal_gamma=rec_gamma,
            recommended_rotation_degrees=rec_rot,
            class_loss_weights=loss_weights,
        )

        fig_paths = self._generate_figures(pair_margins, ar_arr)
        return metrics, fig_paths

    def _generate_figures(
        self,
        pair_margins: List[Tuple[str, str, float]],
        ar_arr: np.ndarray,
    ) -> Dict[str, Path]:
        """Renders publication diagnostic figures for separation margins and aspect ratios.

        Args:
            pair_margins: List of denomination pairings with computed separation margins.
            ar_arr: NumPy array containing bounding box aspect ratio measurements.

        Returns:
            Dictionary mapping plot identifiers to generated filesystem locations.
        """
        fig_paths: Dict[str, Path] = {}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

        top_10 = pair_margins[:10]
        labels = [f"{p[0][:3]} vs {p[1][:3]}" for p in top_10]
        margins = [p[2] for p in top_10]

        sns.barplot(x=margins, y=labels, color="#2b5c8f", ax=axes[0], edgecolor="none")
        axes[0].axvline(0.20, color="#d62728", linestyle="--", linewidth=1.5, label="High Confusion Boundary (< 0.20)")
        EDADesignSystem.apply_theme(
            axes[0],
            title="Denomination Pairwise Separation Margins (Latent Confusion)",
            xlabel="Manifold Separation Margin: Intra - Inter",
            ylabel="Denomination Comparison Pair",
        )
        axes[0].legend(frameon=True, loc="lower right")

        sns.histplot(
            np.clip(ar_arr, 0.2, 3.5),
            bins=35,
            color="#02818a",
            kde=True,
            ax=axes[1],
        )
        axes[1].axvline(0.80, color="#d62728", linestyle="--", linewidth=1.2, label="Vertical Boundary (w/h < 0.80)")
        axes[1].axvline(1.25, color="#2ca02c", linestyle="--", linewidth=1.2, label="Horizontal Boundary (w/h > 1.25)")
        EDADesignSystem.apply_theme(
            axes[1],
            title="Physical Aspect Ratio Orientations Across Banknote Instances",
            xlabel="Aspect Ratio (w / h)",
            ylabel="Annotation Count",
        )
        axes[1].legend(frameon=True, loc="upper right")

        self.paths.figures_eda_dir.mkdir(parents=True, exist_ok=True)
        dive_path_eda = self.paths.figures_eda_dir / "eda_deep_dive_confusions.png"
        EDADesignSystem.save_figure(fig, dive_path_eda)
        fig_paths["deep_dive_confusions"] = dive_path_eda

        return fig_paths
