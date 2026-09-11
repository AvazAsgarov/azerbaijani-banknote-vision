"""Generate 3-panel spatial geometry figure for Slide 5."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Styling constants
CLASS_PALETTE = {
    "001_azn": "#1f77b4",
    "005_azn": "#ff7f0e",
    "010_azn": "#2ca02c",
    "020_azn": "#d62728",
    "050_azn": "#9467bd",
    "100_azn": "#8c564b",
    "200_azn": "#e377c2",
}

def apply_theme(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#1A2B4C")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, fontweight="semibold", labelpad=6, color="#2C3E50")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, fontweight="semibold", labelpad=6, color="#2C3E50")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D5D8DC")
    ax.spines["bottom"].set_color("#D5D8DC")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#D5D8DC")
    ax.set_axisbelow(True)

def main():
    manifest_path = Path("data/processed/dataset_manifest.csv")
    master_labels_dir = Path("data/processed/master/labels")
    alt_labels_dirs = [Path("data/processed/labels/train"), Path("data/processed/labels/val"), Path("data/processed/labels/test")]
    
    df = pd.read_csv(manifest_path)
    if "is_pruned" in df.columns:
        active_df = df[~df["is_pruned"].astype(bool)].copy().reset_index(drop=True)
    else:
        active_df = df.copy().reset_index(drop=True)
        
    print(f"Parsing boxes for {len(active_df)} active images...")
    classes_lookup = [
        "001_azn", "005_azn", "010_azn", "020_azn",
        "050_azn", "100_azn", "200_azn"
    ]
    
    box_records = []
    counts_per_image = []
    
    for _, row in active_df.iterrows():
        fname = str(row["filename"])
        lbl_name = fname.replace(".jpg", ".txt").replace(".png", ".txt")
        lbl_path = master_labels_dir / lbl_name
        if not lbl_path.exists():
            for alt in alt_labels_dirs:
                if (alt / lbl_name).exists():
                    lbl_path = alt / lbl_name
                    break
        
        image_boxes = []
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
                "w": w_safe,
                "h": h_safe,
                "aspect_ratio": ar,
                "rel_area": rel_area,
            })
            
    df_boxes = pd.DataFrame(box_records)
    counts_arr = np.array(counts_per_image)
    print(f"Loaded {len(df_boxes)} boxes across {len(counts_arr)} images.")

    # 3-panel figure layout: 2 on top, 1 centered on bottom
    fig = plt.figure(figsize=(13.5, 9.5), dpi=300)
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.32)
    
    ax_top_left = fig.add_subplot(gs[0, 0:2])
    ax_top_right = fig.add_subplot(gs[0, 2:4])
    ax_bottom_center = fig.add_subplot(gs[1, 1:3])
    
    # 1. Top-Left: Aspect ratio boxplot per class
    order = sorted(list(df_boxes["class_name"].unique()))
    sns.boxplot(
        data=df_boxes,
        x="class_name",
        y="aspect_ratio",
        palette=CLASS_PALETTE,
        order=order,
        ax=ax_top_left,
        fliersize=2,
    )
    ax_top_left.set_ylim(0.2, 3.5)
    ax_top_left.axhline(1.0, color="#d62728", linestyle="--", alpha=0.7, label="Square (1:1)")
    ax_top_left.axhline(1.9, color="#2ca02c", linestyle="--", alpha=0.7, label="Standard Banknote Aspect (~1.9:1)")
    apply_theme(
        ax_top_left,
        title="Bounding Box Aspect Ratio (Width / Height) per Class",
        xlabel="Denomination Class",
        ylabel="Aspect Ratio (w / h)",
    )
    ax_top_left.tick_params(axis="x", rotation=25)
    ax_top_left.legend(frameon=True, loc="upper right", fontsize=8.5)
    
    # 2. Top-Right: Normalized box area distribution (COCO scales)
    sns.histplot(
        df_boxes["rel_area"],
        bins=40,
        color="#2b5c8f",
        kde=True,
        ax=ax_top_right,
    )
    ax_top_right.axvline(0.0025, color="#e66101", linestyle="--", linewidth=1.5, label="Small Object (<0.0025)")
    ax_top_right.axvline(0.0225, color="#02818a", linestyle="--", linewidth=1.5, label="Med/Large (0.0225)")
    apply_theme(
        ax_top_right,
        title="Normalized Bounding Box Area Distribution (COCO Scales)",
        xlabel="Normalized Box Area (w * h / image_area)",
        ylabel="Annotation Count",
    )
    ax_top_right.legend(frameon=True, loc="upper right", fontsize=8.5)
    
    # 3. Bottom-Center: Annotation density
    unique_counts, freq = np.unique(counts_arr, return_counts=True)
    df_counts = pd.DataFrame({"Instances": unique_counts, "Images": freq})
    bars = sns.barplot(
        data=df_counts,
        x="Instances",
        y="Images",
        color="#2b5c8f",
        ax=ax_bottom_center,
        edgecolor="none",
    )
    for p in bars.patches:
        height = p.get_height()
        if height > 0:
            ax_bottom_center.annotate(
                f"{int(height):,} images",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9.5,
                xytext=(0, 4),
                textcoords="offset points",
                fontweight="bold",
                color="#1A2B4C",
            )
    apply_theme(
        ax_bottom_center,
        title="Annotation Density: Objects Detected Per Image",
        xlabel="Number of Banknote Instances in Image",
        ylabel="Image Count",
    )
    ax_bottom_center.set_ylim(0, max(freq) * 1.15)
    
    # Save destinations
    dest1 = Path("reports/figures/eda/eda_spatial_geometry.png")
    dest2 = Path("reports/figures/ai_academy_theme/eda_spatial_geometry.png")
    
    dest1.parent.mkdir(parents=True, exist_ok=True)
    dest2.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(dest1, dpi=300, bbox_inches="tight")
    fig.savefig(dest2, dpi=300, bbox_inches="tight")
    print(f"Successfully exported 3-panel figure to {dest1} and {dest2}")

if __name__ == "__main__":
    main()
