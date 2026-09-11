"""Multi-Objective Combinatorial Group Stratification Engine.

Indivisible meta-scene clusters undergo constrained combinatorial routing across
training, validation, and test subsets. A local search heuristic optimizes minimax
class deviations and overall split ratios while strictly preventing cross-split
background and burst-shot visual leakage.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.core.config import ProjectPaths, SplitConfig
from src.data.clustering import ClusterOutput, MetaSceneClusterer

logger = logging.getLogger("ZeroLeakageSplitter")


@dataclass
class SplitPartitionResult:
    """Quantitative statistical summary of generated dataset partitions.

    Attributes:
        train_count: Total active image count allocated to training partition.
        val_count: Total active image count allocated to validation partition.
        test_count: Total active image count allocated to test partition.
        train_clusters: Set of discrete meta-cluster identifiers in training set.
        val_clusters: Set of discrete meta-cluster identifiers in validation set.
        test_clusters: Set of discrete meta-cluster identifiers in test set.
        class_distribution: DataFrame detailing absolute counts and proportions per class.
    """

    train_count: int
    val_count: int
    test_count: int
    train_clusters: Set[int]
    val_clusters: Set[int]
    test_clusters: Set[int]
    class_distribution: pd.DataFrame


class ZeroLeakageSplitter:
    """Combinatorial multi-objective knapsack partitioner across semantic scene clusters.

    Attributes:
        paths: Configuration paths resolving dataset directories and manifest records.
        split_cfg: Hyperparameters specifying target split ratios and search budgets.
    """

    def __init__(
        self,
        paths: Optional[ProjectPaths] = None,
        split_cfg: Optional[SplitConfig] = None,
    ) -> None:
        """Initializes partitioner with project directory paths and split configuration.

        Args:
            paths: Project configuration paths containing master storage directories.
            split_cfg: Partition configuration specifying ratios and iteration limits.
        """
        self.paths = paths or ProjectPaths()
        self.split_cfg = split_cfg or SplitConfig()

    def _optimize_cluster_assignment(
        self,
        mat: np.ndarray,
        totals: np.ndarray,
        targets: np.ndarray,
        n_groups: int,
        num_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Executes multi-restart local hill-climbing search to find balanced split assignments.

        Minimizes composite objective:
        Loss = 30 * max(diff) + 60 * sum(diff^2) + 10 * sum(diff) + 15 * max(split_err)

        Args:
            mat: Contingency matrix of shape (n_groups, num_classes) with class instance counts.
            totals: Total sample counts across each class of shape (num_classes,).
            targets: Target ratio array [train_ratio, val_ratio, test_ratio].
            n_groups: Number of discrete meta-cluster groups.
            num_classes: Number of distinct denomination categories.

        Returns:
            Tuple of (best_assignment array of group partition indices, best_counts array of shape (3, num_classes)).
        """
        best_score = float("inf")
        best_assignment = np.zeros(n_groups, dtype=int)
        best_counts = np.zeros((3, num_classes))

        np.random.seed(self.split_cfg.random_seed)

        for restart in range(self.split_cfg.optimization_iterations):
            assign = np.random.choice([0, 1, 2], size=n_groups, p=targets)
            step = 0
            improved = True
            while improved and step < 300:
                step += 1
                improved = False
                counts = np.zeros((3, num_classes))
                for s in range(3):
                    counts[s] = np.sum(mat[assign == s], axis=0)
                ratios = counts / (totals + 1e-6)
                diff = np.abs(ratios - targets[:, None])
                tot_samples = np.sum(counts, axis=1)
                tot_err = np.abs(tot_samples / np.sum(totals) - targets)

                score = (
                    np.max(diff) * 30.0 +
                    np.sum(diff**2) * 60.0 +
                    np.sum(diff) * 10.0 +
                    np.max(tot_err) * 15.0
                )

                order = np.random.permutation(n_groups)
                for i in order:
                    orig_s = assign[i]
                    for new_s in [0, 1, 2]:
                        if new_s == orig_s:
                            continue
                        assign[i] = new_s
                        new_counts = np.zeros((3, num_classes))
                        for s in range(3):
                            new_counts[s] = np.sum(mat[assign == s], axis=0)
                        new_ratios = new_counts / (totals + 1e-6)
                        new_diff = np.abs(new_ratios - targets[:, None])
                        new_tot = np.sum(new_counts, axis=1)
                        new_tot_err = np.abs(new_tot / np.sum(totals) - targets)
                        new_score = (
                            np.max(new_diff) * 30.0 +
                            np.sum(new_diff**2) * 60.0 +
                            np.sum(new_diff) * 10.0 +
                            np.max(new_tot_err) * 15.0
                        )
                        if new_score < score:
                            score = new_score
                            improved = True
                            break
                        else:
                            assign[i] = orig_s
                    if improved:
                        break

            if score < best_score:
                best_score = score
                best_assignment = assign.copy()
                final_counts = np.zeros((3, num_classes))
                for s in range(3):
                    final_counts[s] = np.sum(mat[best_assignment == s], axis=0)
                best_counts = final_counts

        return best_assignment, best_counts

    def partition(self, cluster_out: ClusterOutput) -> SplitPartitionResult:
        """Partitions indivisible meta-scene clusters into balanced train, val, and test splits.

        Contingency tables track denomination frequencies across atomic scene communities.
        A multi-restart local hill-climbing search minimizes a composite loss penalizing
        maximum class deviation, squared error, and aggregate split size discrepancy.

        Args:
            cluster_out: Precomputed clustering output containing manifest and meta assignments.

        Returns:
            SplitPartitionResult detailing image counts and cluster memberships.
        """
        df = cluster_out.df_manifest
        classes = sorted(list(self.split_cfg.classes))

        meta_matrix = pd.crosstab(df["meta_cluster"], df["class_name"]).reindex(columns=classes, fill_value=0)
        total_per_class = df["class_name"].value_counts().reindex(classes, fill_value=0)
        meta_clusters = list(meta_matrix.index)

        mat = meta_matrix.values
        totals = meta_matrix.sum(axis=0).values
        n_groups = len(mat)
        targets = np.array([self.split_cfg.train_ratio, self.split_cfg.val_ratio, self.split_cfg.test_ratio])

        best_assignment, best_counts = self._optimize_cluster_assignment(
            mat=mat,
            totals=totals,
            targets=targets,
            n_groups=n_groups,
            num_classes=len(classes),
        )

        tr_c = [meta_clusters[i] for i in range(n_groups) if best_assignment[i] == 0]
        va_c = [meta_clusters[i] for i in range(n_groups) if best_assignment[i] == 1]
        te_c = [meta_clusters[i] for i in range(n_groups) if best_assignment[i] == 2]

        tr_counts = best_counts[0]
        va_counts = best_counts[1]
        te_counts = best_counts[2]

        cluster_to_split: Dict[int, str] = {}
        for c_id in tr_c:
            cluster_to_split[c_id] = "train"
        for c_id in va_c:
            cluster_to_split[c_id] = "val"
        for c_id in te_c:
            cluster_to_split[c_id] = "test"

        df["split"] = df["meta_cluster"].map(cluster_to_split)

        dist_df = pd.DataFrame({
            "class_name": classes,
            "total_count": [total_per_class[c] for c in classes],
            "train_count": [int(tr_counts[i]) for i in range(len(classes))],
            "val_count": [int(va_counts[i]) for i in range(len(classes))],
            "test_count": [int(te_counts[i]) for i in range(len(classes))],
            "train_pct": [f"{tr_counts[i]/total_per_class[c]*100:.1f}%" if total_per_class[c] > 0 else "0.0%" for i, c in enumerate(classes)],
            "val_pct": [f"{va_counts[i]/total_per_class[c]*100:.1f}%" if total_per_class[c] > 0 else "0.0%" for i, c in enumerate(classes)],
            "test_pct": [f"{te_counts[i]/total_per_class[c]*100:.1f}%" if total_per_class[c] > 0 else "0.0%" for i, c in enumerate(classes)],
        })

        self._apply_file_partitioning(df)

        full_df = pd.read_csv(self.paths.manifest_path)
        split_map = dict(zip(df["filename"], df["split"]))
        meta_map = dict(zip(df["filename"], df["meta_cluster"]))
        full_df["split"] = full_df["filename"].map(split_map).fillna("pruned")
        full_df["meta_cluster"] = full_df["filename"].map(meta_map).fillna(-1).astype(int)
        full_df.to_csv(self.paths.manifest_path, index=False)

        logger.info(
            "Optimal split is achieved: Train=%d, Val=%d, Test=%d.",
            int(np.sum(tr_counts)), int(np.sum(va_counts)), int(np.sum(te_counts)),
        )

        return SplitPartitionResult(
            train_count=int(np.sum(tr_counts)),
            val_count=int(np.sum(va_counts)),
            test_count=int(np.sum(te_counts)),
            train_clusters=set(tr_c),
            val_clusters=set(va_c),
            test_clusters=set(te_c),
            class_distribution=dist_df,
        )

    def _apply_file_partitioning(self, df_active: pd.DataFrame) -> None:
        """Synchronizes physical image and label files into target split directories.

        Args:
            df_active: Active dataset manifest containing assigned split designations.
        """
        for split_name in ["train", "val", "test"]:
            img_dir = self.paths.images_dir / split_name
            lbl_dir = self.paths.labels_dir / split_name
            if img_dir.exists():
                shutil.rmtree(img_dir)
            if lbl_dir.exists():
                shutil.rmtree(lbl_dir)
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        master_img_dir = self.paths.master_images_dir
        master_lbl_dir = self.paths.master_labels_dir

        for _, row in df_active.iterrows():
            fname = str(row["filename"])
            lbl_name = fname.replace(".jpg", ".txt").replace(".png", ".txt")
            target_split = str(row["split"])

            src_img = master_img_dir / fname
            src_lbl = master_lbl_dir / lbl_name

            dst_img = self.paths.images_dir / target_split / fname
            dst_lbl = self.paths.labels_dir / target_split / lbl_name

            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    clusterer = MetaSceneClusterer()
    clust_out = clusterer.execute()
    splitter = ZeroLeakageSplitter()
    res = splitter.partition(clust_out)
    print("Zero-leakage split execution is completed.")
    print(res.class_distribution.to_string(index=False))
