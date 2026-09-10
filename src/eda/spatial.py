"""Spatial Geometry and Bounding Annotation Prior Analysis Module.

Bounding box coordinates, aspect ratios, scale distributions, spatial heatmaps,
and optimal anchor configurations are computed to guide detector architecture.
COCO area scale boundaries and K-Means anchor priors inform optimal input
resolutions, feature pyramid strides, and multi-scale detection heads.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from src.core.config import ProjectPaths
from src.eda.style import EDADesignSystem

logger = logging.getLogger("SpatialAnalyzer")


@dataclass
class SpatialMetrics:
    """Quantitative spatial attributes of object bounding annotations.

    Attributes:
        total_annotations: Total quantity of individual bounding boxes parsed.
        mean_aspect_ratio: Global average bounding box aspect ratio (w / h).
        median_aspect_ratio: Median bounding box aspect ratio across all samples.
        aspect_ratio_p05: 5th percentile lower boundary for box aspect ratios.
        aspect_ratio_p95: 95th percentile upper boundary for box aspect ratios.
        pct_small_objects: Percentage of instances with normalized area < 0.0025.
        pct_medium_objects: Percentage of instances with area between 0.0025 and 0.0225.
        pct_large_objects: Percentage of instances with normalized area >= 0.0225.
        pct_single_instance: Percentage of images displaying exactly one banknote.
        pct_multi_instance: Percentage of images displaying multiple banknotes.
        max_instances_per_image: Maximum annotation count occurring within a single frame.
        optimal_anchors_k6: List of (w, h) tuples representing k=6 K-Means anchor priors.
        mean_iou_k6: Mean Intersection over Union between ground truth and optimal anchors.
        recommended_input_resolution: Suggested detector square input resolution.
        recommended_min_stride: Suggested minimal backbone stride for small objects.
    """

    total_annotations: int
    mean_aspect_ratio: float
    median_aspect_ratio: float
    aspect_ratio_p05: float
    aspect_ratio_p95: float
    pct_small_objects: float
    pct_medium_objects: float
    pct_large_objects: float
    pct_single_instance: float
    pct_multi_instance: float
    max_instances_per_image: int
    optimal_anchors_k6: List[Tuple[float, float]]
    mean_iou_k6: float
    recommended_input_resolution: int
    recommended_min_stride: int


class SpatialAnalyzer:
    """Evaluates spatial box geometry and computes optimal anchor allocations.

    Attributes:
        paths: Configuration paths resolving dataset directories and master labels.
    """

    def __init__(self, paths: Optional[ProjectPaths] = None) -> None:
        """Initializes analyzer with standardized filesystem paths.

        Args:
            paths: Optional project configuration paths containing master label directories.
        """
        self.paths = paths or ProjectPaths()

    def _parse_bounding_boxes(self, active_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Parses YOLO spatial bounding annotations from label files across active images.

        Args:
            active_df: Manifest dataframe of active samples.

        Returns:
            Tuple of (DataFrame of parsed box records, array of per-image box counts).
        """
        lbl_master_dir = self.paths.master_labels_dir
        box_records: List[Dict] = []
        counts_per_image: List[int] = []

        classes_lookup = [
            "001_azn", "005_azn", "010_azn", "020_azn",
            "050_azn", "100_azn", "200_azn"
        ]

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

            image_boxes: List[Tuple[int, float, float, float, float]] = []
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:])
                            image_boxes.append((cls_id, xc, yc, w, h))

            counts_per_image.append(len(image_boxes))

            for cls_id, xc, yc, w, h in image_boxes:
                w_safe = max(w, 1e-4)
                h_safe = max(h, 1e-4)
                ar = w_safe / h_safe
                rel_area = w_safe * h_safe
                box_cname = classes_lookup[cls_id] if 0 <= cls_id < len(classes_lookup) else str(row["class_name"])

                box_records.append({
                    "filename": fname,
                    "class_name": box_cname,
                    "split": row["split"],
                    "cls_id": cls_id,
                    "xc": xc,
                    "yc": yc,
                    "w": w_safe,
                    "h": h_safe,
                    "aspect_ratio": ar,
                    "rel_area": rel_area,
                })

        return pd.DataFrame(box_records), np.array(counts_per_image)

    @staticmethod
    def _calculate_kmeans_anchors(wh_data: np.ndarray, k: int = 6) -> Tuple[List[Tuple[float, float]], float]:
        """Calculates optimal anchor box priors using K-Means clustering in width-height space.

        Args:
            wh_data: Normalized bounding box dimensions of shape (N, 2).
            k: Number of anchor clusters to generate.

        Returns:
            Tuple of (ordered list of (width, height) anchor tuples, mean IoU with ground truth).
        """
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(wh_data)
        anchors = sorted(kmeans.cluster_centers_.tolist(), key=lambda x: x[0] * x[1])
        anchors_tuple = [(float(a[0]), float(a[1])) for a in anchors]

        ious: List[float] = []
        for w_gt, h_gt in wh_data:
            box_ious = []
            for a_w, a_h in anchors:
                inter = min(w_gt, a_w) * min(h_gt, a_h)
                union = (w_gt * h_gt) + (a_w * a_h) - inter
                box_ious.append(inter / (union + 1e-6))
            ious.append(max(box_ious))
        mean_anchor_iou = float(np.mean(ious)) if ious else 0.0

        return anchors_tuple, mean_anchor_iou

    def analyze(self) -> Tuple[SpatialMetrics, Dict[str, Path]]:
        """Calculates bounding box proportions, scale classifications, and spatial priors.

        Normalized spatial dimensions assess aspect ratios, COCO scale categorizations,
        and K-Means anchor clusters, providing empirical guidance for detector input resolutions.

        Returns:
            Tuple of calculated SpatialMetrics and generated figure paths.
        """
        df = pd.read_csv(self.paths.manifest_path)
        if "is_pruned" in df.columns:
            active_df = df[~df["is_pruned"].astype(bool)].copy().reset_index(drop=True)
        else:
            active_df = df.copy().reset_index(drop=True)

        logger.info("Parsing YOLO spatial bounding annotations for %d active images...", len(active_df))

        df_boxes, counts_arr = self._parse_bounding_boxes(active_df)
        logger.info("%d total bounding boxes are parsed across %d images.", len(df_boxes), len(counts_arr))

        pct_small = float(np.mean(df_boxes["rel_area"] < 0.0025) * 100.0) if len(df_boxes) > 0 else 0.0
        pct_med = float(np.mean((df_boxes["rel_area"] >= 0.0025) & (df_boxes["rel_area"] < 0.0225)) * 100.0) if len(df_boxes) > 0 else 0.0
        pct_large = float(np.mean(df_boxes["rel_area"] >= 0.0225) * 100.0) if len(df_boxes) > 0 else 0.0

        pct_single = float(np.mean(counts_arr == 1) * 100.0) if len(counts_arr) > 0 else 0.0
        pct_multi = float(np.mean(counts_arr > 1) * 100.0) if len(counts_arr) > 0 else 0.0
        max_instances = int(np.max(counts_arr)) if len(counts_arr) > 0 else 1

        ar_vals = df_boxes["aspect_ratio"].values if len(df_boxes) > 0 else np.array([1.0])
        mean_ar = float(np.mean(ar_vals))
        median_ar = float(np.median(ar_vals))
        p05_ar = float(np.percentile(ar_vals, 5))
        p95_ar = float(np.percentile(ar_vals, 95))

        wh_data = df_boxes[["w", "h"]].values
        anchors_tuple, mean_anchor_iou = self._calculate_kmeans_anchors(wh_data, k=6)
        anchors_list = [[a[0], a[1]] for a in anchors_tuple]

        rec_res = 1024 if pct_small > 5.0 else 640
        rec_stride = 4 if pct_small > 5.0 else 8

        metrics = SpatialMetrics(
            total_annotations=len(df_boxes),
            mean_aspect_ratio=mean_ar,
            median_aspect_ratio=median_ar,
            aspect_ratio_p05=p05_ar,
            aspect_ratio_p95=p95_ar,
            pct_small_objects=pct_small,
            pct_medium_objects=pct_med,
            pct_large_objects=pct_large,
            pct_single_instance=pct_single,
            pct_multi_instance=pct_multi,
            max_instances_per_image=max_instances,
            optimal_anchors_k6=anchors_tuple,
            mean_iou_k6=mean_anchor_iou,
            recommended_input_resolution=rec_res,
            recommended_min_stride=rec_stride,
        )

        fig_paths = self._generate_figures(df_boxes, counts_arr, anchors_list)
        return metrics, fig_paths

    def _generate_figures(
        self,
        df_boxes: pd.DataFrame,
        counts_arr: np.ndarray,
        anchors: List[List[float]],
    ) -> Dict[str, Path]:
        """Renders spatial geometry panels and spatial centroid heatmaps.

        Args:
            df_boxes: DataFrame containing parsed bounding annotation geometry.
            counts_arr: Array recording instance frequencies per image.
            anchors: Cluster centroid list specifying optimal width-height anchors.

        Returns:
            Dictionary mapping plot identifiers to generated filesystem locations.
        """
        fig_paths: Dict[str, Path] = {}
        self.paths.figures_eda_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        order = sorted(list(df_boxes["class_name"].unique()))
        sns.boxplot(
            data=df_boxes,
            x="class_name",
            y="aspect_ratio",
            palette=EDADesignSystem.CLASS_PALETTE,
            order=order,
            ax=axes[0, 0],
            fliersize=2,
        )
        axes[0, 0].set_ylim(0.2, 3.5)
        axes[0, 0].axhline(1.0, color="#d62728", linestyle="--", alpha=0.7, label="Square (1:1)")
        axes[0, 0].axhline(1.9, color="#2ca02c", linestyle="--", alpha=0.7, label="Standard Banknote Aspect (~1.9:1)")
        EDADesignSystem.apply_theme(
            axes[0, 0],
            title="Bounding Box Aspect Ratio (Width / Height) per Class",
            xlabel="Denomination Class",
            ylabel="Aspect Ratio (w / h)",
        )
        axes[0, 0].tick_params(axis="x", rotation=25)
        axes[0, 0].legend(frameon=True, loc="upper right")

        sns.histplot(
            df_boxes["rel_area"],
            bins=40,
            color="#2b5c8f",
            kde=True,
            ax=axes[0, 1],
        )
        axes[0, 1].axvline(0.0025, color="#e66101", linestyle="--", linewidth=1.5, label="Small Object Threshold (<0.0025)")
        axes[0, 1].axvline(0.0225, color="#02818a", linestyle="--", linewidth=1.5, label="Medium/Large Threshold (0.0225)")
        EDADesignSystem.apply_theme(
            axes[0, 1],
            title="Normalized Bounding Box Area Distribution (COCO Scales)",
            xlabel="Normalized Box Area (w * h / image_area)",
            ylabel="Annotation Count",
        )
        axes[0, 1].legend(frameon=True, loc="upper right")

        unique_counts, freq = np.unique(counts_arr, return_counts=True)
        df_counts = pd.DataFrame({"Instances": unique_counts, "Images": freq})
        sns.barplot(
            data=df_counts,
            x="Instances",
            y="Images",
            color="#2b5c8f",
            ax=axes[1, 0],
            edgecolor="none",
        )
        EDADesignSystem.apply_theme(
            axes[1, 0],
            title="Annotation Density: Objects Detected Per Image",
            xlabel="Number of Banknote Instances in Image",
            ylabel="Image Count",
        )

        sns.scatterplot(
            data=df_boxes,
            x="w",
            y="h",
            hue="class_name",
            palette=EDADesignSystem.CLASS_PALETTE,
            alpha=0.35,
            s=15,
            ax=axes[1, 1],
            edgecolor="none",
        )
        anchor_w = [a[0] for a in anchors]
        anchor_h = [a[1] for a in anchors]
        axes[1, 1].scatter(
            anchor_w,
            anchor_h,
            color="#d62728",
            s=120,
            marker="X",
            label="K-Means Anchors (k=6)",
            zorder=10,
            edgecolors="black",
        )
        EDADesignSystem.apply_theme(
            axes[1, 1],
            title="Bounding Box Spatial Dimensions & K-Means Anchors",
            xlabel="Normalized Width (w / W_img)",
            ylabel="Normalized Height (h / H_img)",
        )
        axes[1, 1].legend(loc="upper right", frameon=True, fontsize=8)

        geom_path_eda = self.paths.figures_eda_dir / "eda_spatial_geometry.png"
        EDADesignSystem.save_figure(fig, geom_path_eda)
        fig_paths["spatial_geometry"] = geom_path_eda

        fig, ax = plt.subplots(figsize=(7.5, 7.0))
        sns.kdeplot(
            data=df_boxes,
            x="xc",
            y="yc",
            cmap=EDADesignSystem.SEQUENTIAL_PALETTE,
            fill=True,
            thresh=0.05,
            levels=15,
            ax=ax,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)
        ax.axvline(0.5, color="white", linestyle=":", alpha=0.6)
        ax.axhline(0.5, color="white", linestyle=":", alpha=0.6)
        EDADesignSystem.apply_theme(
            ax,
            title="Spatial Center-of-Mass Density Map (xc, yc)",
            xlabel="Normalized Horizontal Centroid (xc)",
            ylabel="Normalized Vertical Centroid (yc)",
            enable_grid=False,
        )

        heat_path_eda = self.paths.figures_eda_dir / "eda_spatial_heatmap.png"
        EDADesignSystem.save_figure(fig, heat_path_eda)
        fig_paths["spatial_heatmap"] = heat_path_eda

        return fig_paths
