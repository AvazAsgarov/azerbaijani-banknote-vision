"""
Dataset loader and wearable augmentation pipeline for Azerbaijani Banknote detection.
Applies motion blur, optical perspective skew, HSV photometric jitter, and cutout masks.
Pre-caches 160x160 RGB tensors in memory to eliminate disk I/O bottlenecks during training.
Encodes targets into dual-scale anchor grids (Stride 16 and Stride 32) at 160x160 resolution.
"""

import os
import glob
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DENOMINATIONS: List[str] = [
    "001_azn", "005_azn", "010_azn", "020_azn", "050_azn", "100_azn", "200_azn"
]

ANCHORS_S16: np.ndarray = np.array([[118.0, 58.0], [56.0, 112.0], [84.0, 82.0]], dtype=np.float32)
ANCHORS_S32: np.ndarray = np.array([[62.0, 32.0],  [30.0, 60.0],   [40.0, 42.0]], dtype=np.float32)


def compute_anchor_iou(box_wh: Tuple[float, float], anchor_wh: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between bounding box (w, h) and anchor (w, h).

    Args:
        box_wh: Width and height tuple of candidate box in pixel units.
        anchor_wh: Anchor width and height array of shape (2,) in pixel units.

    Returns:
        IoU overlap scalar in range [0.0, 1.0].
    """
    box_area = box_wh[0] * box_wh[1]
    anchor_area = anchor_wh[0] * anchor_wh[1]
    inter_w = min(box_wh[0], anchor_wh[0])
    inter_h = min(box_wh[1], anchor_wh[1])
    inter_area = max(0.0, inter_w * inter_h)
    union_area = box_area + anchor_area - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def apply_motion_blur(image: np.ndarray, kernel_size: int = 5, angle: float = 0.0) -> np.ndarray:
    """Applies directional linear motion blur simulating user head motion.

    Args:
        image: RGB input image array of shape (H, W, 3).
        kernel_size: Linear filter kernel size (must be odd integer).
        angle: Motion trajectory angle in degrees.

    Returns:
        Filtered image array with motion blur applied.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    for i in range(kernel_size):
        offset = i - center
        r = int(round(center + offset * sin_a))
        c = int(round(center + offset * cos_a))
        if 0 <= r < kernel_size and 0 <= c < kernel_size:
            kernel[r, c] = 1.0
    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0
    return cv2.filter2D(image, -1, kernel)


def apply_cutout(image: np.ndarray, num_patches: int = 1, min_size: float = 0.08, max_size: float = 0.20) -> np.ndarray:
    """Simulates user fingers or hands occluding banknote surface with noise patches.

    Args:
        image: Input image array of shape (H, W, 3).
        num_patches: Number of occlusion patches to inject.
        min_size: Minimum patch side dimension relative to image width/height.
        max_size: Maximum patch side dimension relative to image width/height.

    Returns:
        Image array containing random colored occlusion patches.
    """
    h, w = image.shape[:2]
    out = image.copy()
    for _ in range(num_patches):
        pw = int(random.uniform(min_size, max_size) * w)
        ph = int(random.uniform(min_size, max_size) * h)
        px = random.randint(0, max(0, w - pw))
        py = random.randint(0, max(0, h - ph))
        noise = np.random.randint(0, 256, (ph, pw, 3), dtype=np.uint8)
        out[py:py+ph, px:px+pw] = noise
    return out

def crop_and_resize_with_boxes(image, boxes, target_size=160):
    """Crop image to square center and adjust bounding box coordinates.

    Args:
        image: NumPy image array with shape (H, W, 3).
        boxes: List of tuples (class_id, xc, yc, bw, bh) in normalized coordinates.
        target_size: Desired square output dimension.

    Returns:
        Tuple of (resized_square_image, adjusted_boxes).
    """
    h_orig, w_orig = image.shape[:2]
    crop_size = min(h_orig, w_orig)
    x_offset = (w_orig - crop_size) / 2.0
    y_offset = (h_orig - crop_size) / 2.0

    y_start = int(round(y_offset))
    x_start = int(round(x_offset))
    cropped = image[y_start:y_start + crop_size, x_start:x_start + crop_size]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    adjusted = []
    for cls_id, xc, yc, bw, bh in boxes:
        x1 = (xc - bw / 2.0) * w_orig
        y1 = (yc - bh / 2.0) * h_orig
        x2 = (xc + bw / 2.0) * w_orig
        y2 = (yc + bh / 2.0) * h_orig

        x1_c = max(0.0, min(float(crop_size), x1 - x_offset))
        y1_c = max(0.0, min(float(crop_size), y1 - y_offset))
        x2_c = max(0.0, min(float(crop_size), x2 - x_offset))
        y2_c = max(0.0, min(float(crop_size), y2 - y_offset))

        if x2_c > x1_c + 1.0 and y2_c > y1_c + 1.0:
            bw_new = (x2_c - x1_c) / float(crop_size)
            bh_new = (y2_c - y1_c) / float(crop_size)
            xc_new = (x1_c + x2_c) / (2.0 * float(crop_size))
            yc_new = (y1_c + y2_c) / (2.0 * float(crop_size))
            adjusted.append((cls_id, xc_new, yc_new, bw_new, bh_new))

    return resized, adjusted

class BanknoteDataset(Dataset):
    """PyTorch Dataset with RAM caching and wearable augmentations.

    Attributes:
        root_dir: Base directory containing images and labels directories.
        split: Active partition name ('train', 'val', or 'test').
        img_size: Square pixel dimension for model input tensors.
        augment: Boolean enabling motion blur, color jitter, and cutout augmentations.
        samples: List of (image_path, label_path) string tuples.
        cached_images: In-memory cached RGB image arrays if cache_in_ram is enabled.
        cached_boxes: In-memory cached bounding box coordinate lists.
    """

    @staticmethod
    def _load_sample_image_and_boxes(
        img_path: str,
        lbl_path: str,
        img_size: int,
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """Loads and squares an image while synchronizing its YOLO bounding boxes.

        Args:
            img_path: Filesystem path to source image.
            lbl_path: Filesystem path to YOLO annotation text file.
            img_size: Square target dimension for resized image.

        Returns:
            Tuple of (RGB square image array of shape (img_size, img_size, 3), adjusted boxes list).
        """
        img = cv2.imread(img_path)
        boxes: List[Tuple[int, float, float, float, float]] = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        boxes.append((cls_id, xc, yc, bw, bh))

        if img is None:
            img_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            adj_boxes = boxes
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb, adj_boxes = crop_and_resize_with_boxes(img_rgb, boxes, target_size=img_size)

        return img_rgb, adj_boxes

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 160,
        augment: bool = True,
        cache_in_ram: bool = True,
    ) -> None:
        """Initializes dataset loader and optionally pre-caches resized frames in RAM.

        Args:
            root_dir: Dataset root containing images/ and labels/ split folders.
            split: Dataset subset to load ('train', 'val', or 'test').
            img_size: Square input resolution for network.
            augment: When True, applies probabilistic optical augmentations.
            cache_in_ram: When True, pre-loads entire split into RAM.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
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

        print(f"Loaded {len(self.samples)} sample paths for split: {split}", flush=True)

        self.cached_images: List[np.ndarray] = []
        self.cached_boxes: List[List[Tuple[int, float, float, float, float]]] = []
        if cache_in_ram:
            print(f"Pre-caching {len(self.samples)} {split} images at {img_size}x{img_size} in RAM...", flush=True)
            for img_path, lbl_path in self.samples:
                img_rgb, adj_boxes = self._load_sample_image_and_boxes(img_path, lbl_path, img_size)
                self.cached_images.append(img_rgb)
                self.cached_boxes.append(adj_boxes)
            print(f"Caching complete for {split} ({len(self.cached_images)} images stored in RAM).", flush=True)

    def __len__(self) -> int:
        """Returns total registered samples within the dataset split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetches sample tensor and multi-scale encoded ground truth matrices.

        Args:
            idx: Integer index of requested sample.

        Returns:
            Tuple of (normalized input tensor (3, H, W), target_s16, target_s32).
        """
        if self.cached_images:
            image = self.cached_images[idx].copy()
            boxes = list(self.cached_boxes[idx])
        else:
            img_path, lbl_path = self.samples[idx]
            image, boxes = self._load_sample_image_and_boxes(img_path, lbl_path, self.img_size)

        if self.augment:
            if random.random() < 0.35:
                k = random.choice([3, 5])
                ang = random.uniform(-45, 45)
                image = apply_motion_blur(image, kernel_size=k, angle=ang)
            if random.random() < 0.45:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-8, 8)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if random.random() < 0.25:
                image = apply_cutout(image, num_patches=1)

        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        target_s16, target_s32 = self.encode_targets(boxes)
        return img_tensor, target_s16, target_s32

    def encode_targets(
        self,
        boxes: List[Tuple[int, float, float, float, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes raw bounding boxes into training ground-truth matrices for S16 and S32.

        Args:
            boxes: List of tuples (class_id, xc, yc, bw, bh) in normalized coordinates.

        Returns:
            Tuple of (target_s16 tensor of shape (3, 10, 10, 12), target_s32 tensor of shape (3, 5, 5, 12)).
        """
        num_classes = len(DENOMINATIONS)
        target_s16 = np.zeros((3, 10, 10, 5 + num_classes), dtype=np.float32)
        target_s32 = np.zeros((3, 5, 5, 5 + num_classes), dtype=np.float32)

        for cls_id, xc, yc, bw, bh in boxes:
            box_px_w = bw * self.img_size
            box_px_h = bh * self.img_size

            # S16 anchors
            best_iou_16 = -1.0
            best_a_16 = 0
            for a_idx, anch in enumerate(ANCHORS_S16):
                iou = compute_anchor_iou((box_px_w, box_px_h), anch)
                if iou > best_iou_16:
                    best_iou_16 = iou
                    best_a_16 = a_idx

            # S32 anchors
            best_iou_32 = -1.0
            best_a_32 = 0
            for a_idx, anch in enumerate(ANCHORS_S32):
                iou = compute_anchor_iou((box_px_w, box_px_h), anch)
                if iou > best_iou_32:
                    best_iou_32 = iou
                    best_a_32 = a_idx

            if best_iou_16 >= best_iou_32:
                grid_x = min(9, max(0, int(xc * 10)))
                grid_y = min(9, max(0, int(yc * 10)))
                target_s16[best_a_16, grid_y, grid_x, 0] = xc * 10 - grid_x
                target_s16[best_a_16, grid_y, grid_x, 1] = yc * 10 - grid_y
                target_s16[best_a_16, grid_y, grid_x, 2] = np.log(max(1e-4, box_px_w / ANCHORS_S16[best_a_16][0]))
                target_s16[best_a_16, grid_y, grid_x, 3] = np.log(max(1e-4, box_px_h / ANCHORS_S16[best_a_16][1]))
                target_s16[best_a_16, grid_y, grid_x, 4] = 1.0
                if 0 <= cls_id < num_classes:
                    target_s16[best_a_16, grid_y, grid_x, 5 + cls_id] = 1.0
            else:
                grid_x = min(4, max(0, int(xc * 5)))
                grid_y = min(4, max(0, int(yc * 5)))
                target_s32[best_a_32, grid_y, grid_x, 0] = xc * 5 - grid_x
                target_s32[best_a_32, grid_y, grid_x, 1] = yc * 5 - grid_y
                target_s32[best_a_32, grid_y, grid_x, 2] = np.log(max(1e-4, box_px_w / ANCHORS_S32[best_a_32][0]))
                target_s32[best_a_32, grid_y, grid_x, 3] = np.log(max(1e-4, box_px_h / ANCHORS_S32[best_a_32][1]))
                target_s32[best_a_32, grid_y, grid_x, 4] = 1.0
                if 0 <= cls_id < num_classes:
                    target_s32[best_a_32, grid_y, grid_x, 5 + cls_id] = 1.0

        return torch.from_numpy(target_s16), torch.from_numpy(target_s32)
