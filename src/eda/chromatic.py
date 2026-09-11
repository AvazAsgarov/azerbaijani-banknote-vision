"""Foreground Chromaticity and Boundary Truncation Analysis Module.

Extracts ground-truth bounding box crops to isolate banknote currency surfaces
from background textures. Quantifies foreground color profiles in HSV and
CIELAB chromaticity spaces and measures optical canvas boundary truncation rates
to establish rigorous color augmentation boundaries and edge detection safety.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from src.core.config import ProjectPaths
from src.eda.style import EDADesignSystem

logger = logging.getLogger("ChromaticAnalyzer")


@dataclass
class ChromaticMetrics:
    """Quantitative summary of foreground chromatic profiles and boundary margins.

    Attributes:
        class_hue_means: Average HSV Hue angle (0-180 in OpenCV) per denomination.
        class_sat_means: Average HSV Saturation (0-255) per denomination.
        class_val_means: Average HSV Value/Luminance (0-255) per denomination.
        class_truncation_rates: Fraction of boxes touching optical frame edges per class.
        overall_truncation_rate: Global percentage of edge-clipped annotations.
        total_crops_analyzed: Total number of valid ground-truth crops processed.
    """

    class_hue_means: Dict[str, float]
    class_sat_means: Dict[str, float]
    class_val_means: Dict[str, float]
    class_truncation_rates: Dict[str, float]
    overall_truncation_rate: float
    total_crops_analyzed: int


class ForegroundChromaticAnalyzer:
    """Analyzes foreground banknote surface chromaticity and edge proximity.

    Attributes:
        paths: Configuration container resolving dataset and output figure paths.
    """

    def __init__(self, paths: Optional[ProjectPaths] = None) -> None:
        """Initializes the chromatic and boundary analyzer.

        Args:
            paths: ProjectPaths container instance.
        """
        self.paths = paths or ProjectPaths()

    def _extract_crop_color_profile(
        self, img_bgr: np.ndarray, box: List[float]
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Extracts HSV and CIELAB chromaticity means for a single bounding box crop.

        Args:
            img_bgr: Full image array in BGR format.
            box: Normalized bounding coordinates [xc, yc, w, h].

        Returns:
            Tuple of (mean_h, mean_s, mean_v, mean_a, mean_b) or None if invalid.
        """
        h_img, w_img = img_bgr.shape[:2]
        xc, yc, w, h = box

        x1 = max(0, int((xc - w / 2.0) * w_img))
        y1 = max(0, int((yc - h / 2.0) * h_img))
        x2 = min(w_img, int((xc + w / 2.0) * w_img))
        y2 = min(h_img, int((yc + h / 2.0) * h_img))

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        crop_bgr = img_bgr[y1:y2, x1:x2]
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        crop_lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)

        mean_h = float(np.mean(crop_hsv[..., 0]))
        mean_s = float(np.mean(crop_hsv[..., 1]))
        mean_v = float(np.mean(crop_hsv[..., 2]))
        mean_a = float(np.mean(crop_lab[..., 1]))
        mean_b = float(np.mean(crop_lab[..., 2]))

        return mean_h, mean_s, mean_v, mean_a, mean_b

    def _is_truncated(
        self, box: List[float], edge_threshold: float = 0.015
    ) -> Tuple[bool, float]:
        """Determines if a bounding box touches or clips the image boundaries.

        Args:
            box: Normalized bounding coordinates [xc, yc, w, h].
            edge_threshold: Distance threshold to frame border to classify as truncated.

        Returns:
            Tuple of (is_truncated_bool, min_margin_to_edge).
        """
        xc, yc, w, h = box
        x_min = xc - w / 2.0
        x_max = xc + w / 2.0
        y_min = yc - h / 2.0
        y_max = yc + h / 2.0

        min_margin = min(x_min, y_min, 1.0 - x_max, 1.0 - y_max)
        is_touching = (
            x_min <= edge_threshold
            or y_min <= edge_threshold
            or x_max >= (1.0 - edge_threshold)
            or y_max >= (1.0 - edge_threshold)
        )
        return is_touching, float(max(0.0, min_margin))

    def analyze(self) -> Tuple[ChromaticMetrics, Dict[str, Path]]:
        """Executes full foreground extraction, color profiling, and boundary analysis.

        Returns:
            Tuple containing calculated ChromaticMetrics and a dictionary of saved figure paths.
        """
        logger.info("Parsing active dataset images for foreground chromatic and boundary analysis...")
        manifest_path = self.paths.manifest_path

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

        df_manifest = pd.read_csv(manifest_path)
        if "is_pruned" in df_manifest.columns:
            active_df = df_manifest[~df_manifest["is_pruned"].astype(bool)].copy()
        else:
            active_df = df_manifest.copy()

        crop_records = []
        class_names = [
            "001_azn", "005_azn", "010_azn", "020_azn", "050_azn", "100_azn", "200_azn"
        ]

        for _, row in tqdm(active_df.iterrows(), total=len(active_df), desc="Analyzing foregrounds"):
            split = str(row.get("split", "train"))
            filename = str(row["filename"])
            stem = Path(filename).stem

            # Try locating label file
            label_path = self.paths.labels_dir / split / f"{stem}.txt"
            if not label_path.exists():
                label_path = self.paths.labels_dir / f"{stem}.txt"

            # Try locating image file
            img_path = self.paths.images_dir / split / filename
            if not img_path.exists():
                img_path = self.paths.images_dir / filename
            if not img_path.exists():
                img_path = self.paths.master_images_dir / filename

            if not img_path.exists() or not label_path.exists():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                if cls_idx >= len(class_names):
                    continue
                cls_name = class_names[cls_idx]
                box = [float(p) for p in parts[1:]]

                color_profile = self._extract_crop_color_profile(img_bgr, box)
                if color_profile is None:
                    continue

                is_trunc, margin = self._is_truncated(box)
                h, s, v, lab_a, lab_b = color_profile

                crop_records.append({
                    "filename": filename,
                    "split": split,
                    "class_name": cls_name,
                    "hue": h,
                    "saturation": s,
                    "value": v,
                    "lab_a": lab_a,
                    "lab_b": lab_b,
                    "is_truncated": is_trunc,
                    "min_margin": margin,
                    "box_width": box[2],
                    "box_height": box[3],
                })

        df_crops = pd.DataFrame(crop_records)
        if df_crops.empty:
            raise ValueError("No valid foreground banknote crops could be extracted.")

        # Compute summary metrics
        hue_means = df_crops.groupby("class_name")["hue"].mean().to_dict()
        sat_means = df_crops.groupby("class_name")["saturation"].mean().to_dict()
        val_means = df_crops.groupby("class_name")["value"].mean().to_dict()
        trunc_rates = df_crops.groupby("class_name")["is_truncated"].mean().to_dict()
        overall_trunc = float(df_crops["is_truncated"].mean())

        metrics = ChromaticMetrics(
            class_hue_means={k: round(float(v), 2) for k, v in hue_means.items()},
            class_sat_means={k: round(float(v), 2) for k, v in sat_means.items()},
            class_val_means={k: round(float(v), 2) for k, v in val_means.items()},
            class_truncation_rates={k: round(float(v), 4) for k, v in trunc_rates.items()},
            overall_truncation_rate=round(overall_trunc, 4),
            total_crops_analyzed=len(df_crops),
        )

        fig_paths = self._render_figures(df_crops, metrics)
        return metrics, fig_paths

    def _render_figures(
        self, df_crops: pd.DataFrame, metrics: ChromaticMetrics
    ) -> Dict[str, Path]:
        """Generates publication-grade figures for chromaticity and boundary truncation.

        Args:
            df_crops: DataFrame containing crop-level chromatic and boundary metrics.
            metrics: ChromaticMetrics summary container.

        Returns:
            Dictionary mapping plot names to generated filesystem paths.
        """
        fig_paths: Dict[str, Path] = {}

        # ---------------------------------------------------------------------
        # Figure 1: Foreground Banknote Chromaticity Profile (HSV Hue vs Saturation)
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Subplot 1: Hue vs Saturation Scatter per Denomination
        for cls_name, color in EDADesignSystem.CLASS_PALETTE.items():
            sub = df_crops[df_crops["class_name"] == cls_name]
            if not sub.empty:
                # Subsample if large for crisp plotting
                n_pts = min(len(sub), 200)
                sub_sample = sub.sample(n=n_pts, random_state=42)
                axes[0].scatter(
                    sub_sample["hue"],
                    sub_sample["saturation"],
                    label=cls_name,
                    color=color,
                    alpha=0.6,
                    s=28,
                    edgecolors="none",
                )

        EDADesignSystem.apply_theme(
            axes[0],
            title="Foreground Banknote Chromaticity Distribution (HSV)",
            xlabel="HSV Hue Channel (OpenCV scale 0-180)",
            ylabel="HSV Saturation Channel (0-255)",
        )
        axes[0].legend(title="Denomination", loc="upper right", frameon=True, fontsize=8)

        # Subplot 2: CIELAB a* (Green-Red) vs b* (Blue-Yellow) Chromaticity
        for cls_name, color in EDADesignSystem.CLASS_PALETTE.items():
            sub = df_crops[df_crops["class_name"] == cls_name]
            if not sub.empty:
                n_pts = min(len(sub), 200)
                sub_sample = sub.sample(n=n_pts, random_state=42)
                axes[1].scatter(
                    sub_sample["lab_a"],
                    sub_sample["lab_b"],
                    label=cls_name,
                    color=color,
                    alpha=0.6,
                    s=28,
                    edgecolors="none",
                )

        axes[1].axvline(128, color="gray", linestyle=":", alpha=0.5)
        axes[1].axhline(128, color="gray", linestyle=":", alpha=0.5)
        EDADesignSystem.apply_theme(
            axes[1],
            title="CIELAB Chromaticity Space (a* vs b*)",
            xlabel="CIELAB a* Channel (Green <- 128 -> Red)",
            ylabel="CIELAB b* Channel (Blue <- 128 -> Yellow)",
        )
        axes[1].legend(title="Denomination", loc="upper right", frameon=True, fontsize=8)

        chroma_path = self.paths.figures_eda_dir / "eda_foreground_chromaticity.png"
        EDADesignSystem.save_figure(fig, chroma_path)
        fig_paths["foreground_chromaticity"] = chroma_path

        # ---------------------------------------------------------------------
        # Figure 2: Boundary Truncation Frequency & Margin to Frame Edge
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

        # Subplot 1: Truncation Rate per Denomination
        class_order = list(EDADesignSystem.CLASS_PALETTE.keys())
        rates_pct = [metrics.class_truncation_rates.get(c, 0.0) * 100 for c in class_order]
        colors = [EDADesignSystem.CLASS_PALETTE[c] for c in class_order]

        bars = axes[0].barh(class_order, rates_pct, color=colors, height=0.6, edgecolor="none")
        axes[0].axvline(
            metrics.overall_truncation_rate * 100,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label=f"Dataset Mean ({metrics.overall_truncation_rate * 100:.1f}%)"
        )
        for bar in bars:
            w = bar.get_width()
            axes[0].text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5)

        axes[0].set_xlim(0, max(rates_pct + [20.0]) + 8)
        EDADesignSystem.apply_theme(
            axes[0],
            title="Boundary Truncation & Edge-Clipping Frequency",
            xlabel="Truncated / Edge-Adjacent Annotations (%)",
            ylabel="Denomination Class",
        )
        axes[0].legend(loc="lower right", frameon=True, fontsize=9)

        # Subplot 2: Minimum Distance Margin to Image Edge
        sns.histplot(
            data=df_crops,
            x="min_margin",
            hue="split",
            palette=EDADesignSystem.SPLIT_PALETTE,
            bins=35,
            element="step",
            common_norm=False,
            stat="density",
            ax=axes[1],
        )
        axes[1].axvline(0.015, color="#d62728", linestyle=":", linewidth=1.5, label="Boundary Threshold (1.5%)")
        EDADesignSystem.apply_theme(
            axes[1],
            title="Spatial Margin Distribution to Frame Boundary",
            xlabel="Normalized Margin to Closest Canvas Edge",
            ylabel="Density Distribution",
        )
        axes[1].legend(title="Split", loc="upper right", frameon=True, fontsize=8.5)

        trunc_path = self.paths.figures_eda_dir / "eda_boundary_truncation.png"
        EDADesignSystem.save_figure(fig, trunc_path)
        fig_paths["boundary_truncation"] = trunc_path

        return fig_paths
