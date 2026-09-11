"""FOMO Dataset and Loss utilities for Centroid Detection.

Encodes bounding box annotations into 1-hot / categorical centroid grids.
Class 0: Background
Class 1..7: Azerbaijani Banknote denominations (001_azn .. 200_azn)
"""

import os
import glob
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.tinyml.dataset import DENOMINATIONS, crop_and_resize_with_boxes


class FOMOBanknoteDataset(Dataset):
    """Dataset for FOMO Centroid Heatmap training."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 96,
        grid_size: int = 12,
        augment: bool = True,
        cache_in_ram: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.grid_size = grid_size
        self.augment = augment

        img_dir = os.path.join(root_dir, "images", split)
        lbl_dir = os.path.join(root_dir, "labels", split)

        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.samples: List[Tuple[str, str]] = []
        for img_path in all_imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lbl_path):
                self.samples.append((img_path, lbl_path))

        self.cached_images: List[np.ndarray] = []
        self.cached_targets: List[np.ndarray] = []

        if cache_in_ram:
            print(f"Pre-caching {len(self.samples)} {split} images for FOMO ({img_size}x{img_size})...", flush=True)
            for img_p, lbl_p in self.samples:
                img_rgb, boxes = self._load_sample(img_p, lbl_p, img_size)
                target_grid = self._encode_centroids(boxes, grid_size)
                self.cached_images.append(img_rgb)
                self.cached_targets.append(target_grid)
            print(f"FOMO caching complete for {split} ({len(self.cached_images)} images).", flush=True)

    def _load_sample(self, img_path: str, lbl_path: str, img_size: int) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        boxes = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((c, x, y, w, h))

        img = cv2.imread(img_path)
        if img is None:
            img_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            adj_boxes = boxes
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb, adj_boxes = crop_and_resize_with_boxes(img_rgb, boxes, target_size=img_size)

        return img_rgb, adj_boxes

    def _encode_centroids(self, boxes: List[Tuple[int, float, float, float, float]], grid_size: int) -> np.ndarray:
        # Target grid: (H_g, W_g) with class indices 0..7 (0 is background)
        grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        for (c, xc, yc, bw, bh) in boxes:
            gx = int(min(grid_size - 1, max(0, math.floor(xc * grid_size))))
            gy = int(min(grid_size - 1, max(0, math.floor(yc * grid_size))))
            # Label is c + 1 (1 to 7)
            grid[gy, gx] = c + 1
        return grid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cached_images:
            img = self.cached_images[idx].copy()
            target = self.cached_targets[idx].copy()
        else:
            img_p, lbl_p = self.samples[idx]
            img, boxes = self._load_sample(img_p, lbl_p, self.img_size)
            target = self._encode_centroids(boxes, self.grid_size)

        # Basic light photometric jitter if train augment
        if self.augment and np.random.rand() > 0.5:
            delta = np.random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        target_tensor = torch.from_numpy(target).long()
        return img_tensor, target_tensor


class FOMOFocalLoss(nn.Module):
    """Focal Loss for FOMO Centroid Detection to address heavy background class imbalance."""

    def __init__(self, num_classes: int = 7, gamma: float = 2.0, alpha_bg: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        # Background class gets weight alpha_bg, currency classes get 1.0
        weights = torch.ones(num_classes + 1)
        weights[0] = alpha_bg
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Logits: (B, num_classes + 1, H, W), Targets: (B, H, W)."""
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
