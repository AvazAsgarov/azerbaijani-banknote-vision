"""Photometric Signal Processing and Augmentation Boundary Analysis Module.

Luminance shifts, channel histograms, contrast ratios, blur profiles,
and frequency domain properties are quantified to establish augmentation bounds.
Spectral analysis via Fast Fourier Transforms and Laplacian focus operators
prevents out-of-distribution artifacts during downstream detector training.
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

logger = logging.getLogger("PhotometricAnalyzer")


@dataclass
class PhotometricMetrics:
    """Quantitative photometric and frequency characteristics across the image pool.

    Attributes:
        mean_luminance: Global average grayscale pixel intensity (0 to 255 scale).
        std_luminance: Global standard deviation of grayscale intensities.
        luminance_p05: 5th percentile lower boundary for image brightness.
        luminance_p95: 95th percentile upper boundary for image brightness.
        mean_rgb: Average channel intensities for red, green, and blue components.
        std_rgb: Channel intensity standard deviations across RGB channels.
        mean_rms_contrast: Average Root-Mean-Square contrast across all samples.
        median_laplacian_var: Median focus measure calculated via Laplacian variance.
        pct_sharp_frames: Percentage of images classified as sharp (variance >= 100).
        pct_moderate_blur: Percentage of images exhibiting mild blur (variance 30 to 100).
        pct_heavy_blur: Percentage of images exhibiting heavy blur (variance < 30).
        recommended_brightness_delta: Maximum brightness jitter range for data augmentation.
        recommended_contrast_delta: Maximum contrast jitter range for data augmentation.
        recommended_saturation_delta: Maximum color saturation jitter range for augmentation.
        recommended_hue_delta: Maximum hue perturbation delta for data augmentation.
        recommended_blur_kernel_max: Maximum Gaussian blur kernel dimension for augmentation.
    """

    mean_luminance: float
    std_luminance: float
    luminance_p05: float
    luminance_p95: float
    mean_rgb: Tuple[float, float, float]
    std_rgb: Tuple[float, float, float]
    mean_rms_contrast: float
    median_laplacian_var: float
    pct_sharp_frames: float
    pct_moderate_blur: float
    pct_heavy_blur: float
    recommended_brightness_delta: float
    recommended_contrast_delta: float
    recommended_saturation_delta: float
    recommended_hue_delta: float
    recommended_blur_kernel_max: int


class PhotometricAnalyzer:
    """Extracts pixel-level signal metrics and establishes defensible augmentation limits.

    Attributes:
        paths: Configuration paths resolving dataset directories and master image folders.
    """

    def __init__(self, paths: Optional[ProjectPaths] = None) -> None:
        """Initializes analyzer with standardized filesystem paths.

        Args:
            paths: Optional project configuration paths containing master image directories.
        """
        self.paths = paths or ProjectPaths()

    @staticmethod
    def _extract_single_image_photometrics(img_path: Path) -> Optional[Dict[str, float]]:
        """Extracts pixel luminance, RGB channel balance, contrast, FFT ratio, and blur metrics for one frame.

        Args:
            img_path: Absolute or relative Path pointing to the target image file.

        Returns:
            Dictionary of scalar photometric indicators, or None if image reading fails.
        """
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        lum_mean = float(np.mean(gray))
        lum_std = float(np.std(gray))
        r_mean = float(np.mean(img_rgb[:, :, 0]))
        g_mean = float(np.mean(img_rgb[:, :, 1]))
        b_mean = float(np.mean(img_rgb[:, :, 2]))

        rms = float(lum_std / (lum_mean + 1e-6))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        gray_std = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        f_transform = np.fft.fft2(gray_std)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        cy, cx = 256, 256
        y, x = np.ogrid[:512, :512]
        radius = 51.2
        mask_low = ((x - cx)**2 + (y - cy)**2) <= radius**2
        total_energy = np.sum(magnitude) + 1e-6
        high_freq_energy = np.sum(magnitude[~mask_low])
        fft_ratio = float(high_freq_energy / total_energy)

        blurred = cv2.medianBlur(gray_std, 3)
        residual = gray_std.astype(np.float32) - blurred.astype(np.float32)
        noise_sigma = float(np.std(residual))

        return {
            "luminance": lum_mean,
            "lum_std": lum_std,
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
            "rms_contrast": rms,
            "laplacian_var": lap_var,
            "fft_ratio": fft_ratio,
            "noise_std": noise_sigma,
        }

    def analyze(self) -> Tuple[PhotometricMetrics, Dict[str, Path]]:
        """Calculates pixel luminance, channel balance, contrast, and sharpness metrics.

        Fast Fourier Transform spectral energy ratios and Laplacian focus measures assess
        high-frequency characteristics, establishing empirical bounds for data augmentation.

        Returns:
            Tuple of calculated PhotometricMetrics and generated figure paths.
        """
        df = pd.read_csv(self.paths.manifest_path)
        if "is_pruned" in df.columns:
            active_df = df[~df["is_pruned"].astype(bool)].copy().reset_index(drop=True)
        else:
            active_df = df.copy().reset_index(drop=True)

        img_dir = self.paths.master_images_dir

        logger.info("Extracting photometric signatures across %d active images...", len(active_df))

        luminances: List[float] = []
        r_means: List[float] = []
        g_means: List[float] = []
        b_means: List[float] = []
        rms_contrasts: List[float] = []
        laplacian_vars: List[float] = []
        fft_energy_ratios: List[float] = []
        noise_stds: List[float] = []

        valid_rows: List[Dict] = []

        for idx, row in tqdm(active_df.iterrows(), total=len(active_df), desc="Photometric Analysis"):
            fname = str(row["filename"])
            img_path = img_dir / fname
            if not img_path.exists():
                for s in ["train", "val", "test"]:
                    alt = self.paths.images_dir / s / fname
                    if alt.exists():
                        img_path = alt
                        break

            metrics_dict = self._extract_single_image_photometrics(img_path)
            if metrics_dict is None:
                continue

            luminances.append(metrics_dict["luminance"])
            r_means.append(metrics_dict["r_mean"])
            g_means.append(metrics_dict["g_mean"])
            b_means.append(metrics_dict["b_mean"])
            rms_contrasts.append(metrics_dict["rms_contrast"])
            laplacian_vars.append(metrics_dict["laplacian_var"])
            fft_energy_ratios.append(metrics_dict["fft_ratio"])
            noise_stds.append(metrics_dict["noise_std"])

            valid_rows.append({
                "filename": fname,
                "class_name": row["class_name"],
                "split": row["split"],
                **metrics_dict,
            })

        df_metrics = pd.DataFrame(valid_rows)

        p05 = float(np.percentile(luminances, 5))
        p95 = float(np.percentile(luminances, 95))
        mean_lum = float(np.mean(luminances))
        std_lum = float(np.std(luminances))

        pct_sharp = float(np.mean([v >= 100.0 for v in laplacian_vars])) * 100.0
        pct_moderate = float(np.mean([(v >= 30.0 and v < 100.0) for v in laplacian_vars])) * 100.0
        pct_heavy = float(np.mean([v < 30.0 for v in laplacian_vars])) * 100.0

        rec_bright = float(min(0.30, max(0.15, std_lum / mean_lum * 0.5)))
        rec_contrast = float(min(0.30, max(0.15, np.std(rms_contrasts) * 1.5)))
        rec_saturation = 0.25
        rec_hue = 0.05
        rec_blur_k = 5 if pct_heavy < 10.0 else 3

        metrics = PhotometricMetrics(
            mean_luminance=mean_lum,
            std_luminance=std_lum,
            luminance_p05=p05,
            luminance_p95=p95,
            mean_rgb=(float(np.mean(r_means)), float(np.mean(g_means)), float(np.mean(b_means))),
            std_rgb=(float(np.std(r_means)), float(np.std(g_means)), float(np.std(b_means))),
            mean_rms_contrast=float(np.mean(rms_contrasts)),
            median_laplacian_var=float(np.median(laplacian_vars)),
            pct_sharp_frames=pct_sharp,
            pct_moderate_blur=pct_moderate,
            pct_heavy_blur=pct_heavy,
            recommended_brightness_delta=rec_bright,
            recommended_contrast_delta=rec_contrast,
            recommended_saturation_delta=rec_saturation,
            recommended_hue_delta=rec_hue,
            recommended_blur_kernel_max=rec_blur_k,
        )

        fig_paths = self._generate_figures(df_metrics)
        return metrics, fig_paths

    def _generate_figures(self, df_metrics: pd.DataFrame) -> Dict[str, Path]:
        """Renders multi-panel photometric profiles and frequency sharpness diagnostic figures.

        Args:
            df_metrics: DataFrame containing per-sample photometric and frequency measurements.

        Returns:
            Dictionary mapping plot identifiers to generated filesystem locations.
        """
        fig_paths: Dict[str, Path] = {}
        self.paths.figures_eda_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sns.kdeplot(
            data=df_metrics,
            x="luminance",
            hue="split",
            palette=EDADesignSystem.SPLIT_PALETTE,
            fill=True,
            alpha=0.3,
            common_norm=False,
            ax=axes[0, 0],
        )
        EDADesignSystem.apply_theme(
            axes[0, 0],
            title="Luminance Distribution across Dataset Splits",
            xlabel="Mean Grayscale Intensity (0-255)",
            ylabel="Probability Density",
        )

        sns.kdeplot(df_metrics["r_mean"], color="#d62728", label="Red Channel", fill=True, alpha=0.25, ax=axes[0, 1])
        sns.kdeplot(df_metrics["g_mean"], color="#2ca02c", label="Green Channel", fill=True, alpha=0.25, ax=axes[0, 1])
        sns.kdeplot(df_metrics["b_mean"], color="#1f77b4", label="Blue Channel", fill=True, alpha=0.25, ax=axes[0, 1])
        EDADesignSystem.apply_theme(
            axes[0, 1],
            title="RGB Spectral Color Channel Densities",
            xlabel="Pixel Intensity",
            ylabel="Probability Density",
        )
        axes[0, 1].legend(frameon=True, loc="upper right")

        order = sorted(list(df_metrics["class_name"].unique()))
        sns.boxplot(
            data=df_metrics,
            x="class_name",
            y="rms_contrast",
            palette=EDADesignSystem.CLASS_PALETTE,
            order=order,
            ax=axes[1, 0],
            fliersize=2,
        )
        EDADesignSystem.apply_theme(
            axes[1, 0],
            title="RMS Contrast Variability across Currency Denominations",
            xlabel="Denomination Class",
            ylabel="RMS Contrast (std / mean)",
        )
        axes[1, 0].tick_params(axis="x", rotation=25)

        sns.histplot(
            data=df_metrics,
            x="noise_std",
            bins=35,
            color="#2b5c8f",
            kde=True,
            ax=axes[1, 1],
        )
        EDADesignSystem.apply_theme(
            axes[1, 1],
            title="Sensor Noise Residual Standard Deviation",
            xlabel="Residual Sigma (Median Filter Difference)",
            ylabel="Sample Count",
        )

        dist_path_eda = self.paths.figures_eda_dir / "eda_photometric_distributions.png"
        EDADesignSystem.save_figure(fig, dist_path_eda)
        fig_paths["photometric_distributions"] = dist_path_eda

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

        log_lap = np.log10(np.clip(df_metrics["laplacian_var"], 1.0, 1e6))
        sns.histplot(log_lap, bins=35, color="#02818a", kde=True, ax=axes[0])
        axes[0].axvline(np.log10(30.0), color="#d62728", linestyle="--", linewidth=1.5, label="Heavy Blur Limit (Var < 30)")
        axes[0].axvline(np.log10(100.0), color="#2ca02c", linestyle="--", linewidth=1.5, label="Sharp Transition (Var > 100)")
        EDADesignSystem.apply_theme(
            axes[0],
            title="Laplacian Focus Measure (Sharpness Distribution)",
            xlabel="log10(Laplacian Variance)",
            ylabel="Image Count",
        )
        axes[0].legend(frameon=True, loc="upper left", fontsize=9)

        sns.scatterplot(
            data=df_metrics,
            x="fft_ratio",
            y=log_lap,
            hue="split",
            palette=EDADesignSystem.SPLIT_PALETTE,
            alpha=0.5,
            s=20,
            ax=axes[1],
            edgecolor="none",
        )
        EDADesignSystem.apply_theme(
            axes[1],
            title="Frequency Spectral Energy vs Spatial Sharpness",
            xlabel="High-Frequency Energy Ratio (FFT)",
            ylabel="log10(Laplacian Variance)",
        )
        axes[1].legend(title="Split", loc="lower right", frameon=True)

        sharp_path_eda = self.paths.figures_eda_dir / "eda_sharpness_frequency.png"
        EDADesignSystem.save_figure(fig, sharp_path_eda)
        fig_paths["sharpness_frequency"] = sharp_path_eda

        return fig_paths
