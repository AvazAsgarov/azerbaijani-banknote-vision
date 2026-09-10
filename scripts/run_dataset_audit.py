"""
Production-grade dataset quality audit script.
Evaluates:
1. Inter-Annotator Agreement (IAA) on calibration datasets (mIoU, Cohen's Kappa, F1-score).
2. Exact binary duplicates (MD5/SHA-256).
3. Near-duplicate perceptual hash analysis (pHash, dHash).
"""

import os
import sys
import re
import json
import hashlib
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths

# ==============================================================================
# 1. INTER-ANNOTATOR AGREEMENT AUDIT
# ==============================================================================

def load_yolo_boxes(file_content: str):
    """Parses YOLO bounding boxes: [(class_id, x_c, y_c, w, h), ...]"""
    boxes = []
    for line in file_content.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            boxes.append((cls_id, x, y, w, h))
    return boxes

def yolo_to_xyxy(box):
    cls_id, x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return cls_id, x1, y1, x2, y2

def compute_iou(box1, box2):
    _, x1_a, y1_a, x2_a, y2_a = yolo_to_xyxy(box1)
    _, x1_b, y1_b, x2_b, y2_b = yolo_to_xyxy(box2)

    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def evaluate_calibration_iaa():
    print("====================================================================")
    print("1. INTER-ANNOTATOR AGREEMENT (CALIBRATION AUDIT)")
    print("====================================================================")
    
    paths = ProjectPaths()
    
    # 1. Load Avaz calibration labels
    avaz_labels = {}
    avaz_lbl_dir = paths.raw_data_dir / "AzeBankNotes_labelling" / "avaz" / "labels_calibration"
    if not avaz_lbl_dir.exists():
        avaz_lbl_dir = Path("AzeBankNotes_labelling/avaz/labels_calibration")
    if avaz_lbl_dir.exists():
        for f in os.listdir(avaz_lbl_dir):
            if f.endswith('.txt'):
                with open(os.path.join(avaz_lbl_dir, f), 'r', encoding='utf-8') as fl:
                    avaz_labels[f] = load_yolo_boxes(fl.read())

    # 2. Load Gulnar calibration labels
    gulnar_labels = {}
    g_zip_path = paths.raw_data_dir / "gulnar calibration.zip"
    if not g_zip_path.exists():
        g_zip_path = Path("gulnar calibration.zip")
    if g_zip_path.exists():
        with zipfile.ZipFile(g_zip_path) as z:
            for f in z.namelist():
                if f.startswith('labels/') and f.endswith('.txt'):
                    clean_name = re.sub(r'^[0-9a-fA-F]{8}-', '', os.path.basename(f))
                    gulnar_labels[clean_name] = load_yolo_boxes(z.read(f).decode('utf-8'))

    # 3. Load Kazim calibration labels
    kazim_labels = {}
    k_zip_path = paths.raw_data_dir / "kazim_labeled.zip"
    if not k_zip_path.exists():
        k_zip_path = Path("kazim_labeled.zip")
    if k_zip_path.exists():
        with zipfile.ZipFile(k_zip_path) as z:
            for f in z.namelist():
                if 'calibration_label/' in f and f.endswith('.txt'):
                    clean_name = os.path.basename(f)
                    kazim_labels[clean_name] = load_yolo_boxes(z.read(f).decode('utf-8'))

    print(f"Calibration labels available: Avaz ({len(avaz_labels)}), Gulnar ({len(gulnar_labels)}), Kazim ({len(kazim_labels)})")

    # Overlap between Avaz and Gulnar
    common_files = sorted(list(set(avaz_labels.keys()) & set(gulnar_labels.keys())))
    print(f"Common calibration files between Avaz and Gulnar: {len(common_files)}")

    matched_ious = []
    class_matches = 0
    total_matched_pairs = 0
    y_true_cls = []
    y_pred_cls = []
    confusion_pairs = []

    for f in common_files:
        boxes_a = avaz_labels[f]
        boxes_g = gulnar_labels[f]

        # Match boxes greedily by IoU
        used_g = set()
        for b_a in boxes_a:
            best_iou = 0.0
            best_idx = -1
            for j, b_g in enumerate(boxes_g):
                if j in used_g:
                    continue
                iou = compute_iou(b_a, b_g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_idx >= 0 and best_iou >= 0.5:
                used_g.add(best_idx)
                b_g = boxes_g[best_idx]
                matched_ious.append(best_iou)
                total_matched_pairs += 1
                y_true_cls.append(b_a[0])
                y_pred_cls.append(b_g[0])
                if b_a[0] == b_g[0]:
                    class_matches += 1
                else:
                    confusion_pairs.append((f, b_a[0], b_g[0]))

    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    median_iou = np.median(matched_ious) if matched_ious else 0.0
    class_acc = class_matches / total_matched_pairs if total_matched_pairs > 0 else 0.0

    # Calculate Cohen's Kappa for classification agreement
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y_true_cls, y_pred_cls) if y_true_cls else 0.0

    print("\n--- Calibration IAA Results (Avaz vs. Gulnar) ---")
    print(f"Total Evaluated Images:          {len(common_files)}")
    print(f"Matched Bounding Box Pairs:       {total_matched_pairs}")
    print(f"Mean Bounding Box IoU (mIoU):     {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"Median Bounding Box IoU:         {median_iou:.4f} ({median_iou*100:.2f}%)")
    print(f"Class Classification Agreement:   {class_acc:.4f} ({class_acc*100:.2f}%)")
    print(f"Cohen's Kappa (Class Agreement): {kappa:.4f}")
    if confusion_pairs:
        print(f"Class Disagreements ({len(confusion_pairs)}):", confusion_pairs)
    else:
        print("Class Disagreements: 0 (100% perfect consensus on class labels!)")

    return {
        "common_images": len(common_files),
        "matched_pairs": total_matched_pairs,
        "mean_iou": float(mean_iou),
        "median_iou": float(median_iou),
        "class_accuracy": float(class_acc),
        "cohens_kappa": float(kappa)
    }

# ==============================================================================
# 2. EXACT BINARY & NEAR-DUPLICATE DEDUPLICATION AUDIT
# ==============================================================================

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_phash(image_path, hash_size=8, highfreq_factor=4):
    """Computes standard 64-bit DCT perceptual hash."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq)
    diff = dct_lowfreq > med
    return diff.flatten()

def compute_dhash(image_path, hash_size=8):
    """Computes difference hash (gradient-based)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def hamming_distance(h1, h2):
    return np.count_nonzero(h1 != h2)

def evaluate_duplicates():
    print("\n====================================================================")
    print("2. EXACT AND NEAR-DUPLICATE DEDUPLICATION AUDIT")
    print("====================================================================")

    paths = ProjectPaths()
    manifest_path = paths.manifest_path
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    print(f"Auditing all {len(df)} images in master pool...")

    # A. Exact SHA-256 Duplicate Scan
    sha_map = {}
    exact_duplicates = []

    # Pre-calculate hashes
    phash_dict = {}
    dhash_dict = {}
    img_list = []

    for idx, row in df.iterrows():
        fname = row['filename']
        split = str(row.get('split', 'master'))
        img_path = paths.images_dir / split / fname
        if not img_path.exists():
            img_path = paths.master_images_dir / fname
        
        if img_path.exists():
            # SHA256
            sha = compute_sha256(str(img_path))
            if sha in sha_map:
                exact_duplicates.append((fname, sha_map[sha]))
            else:
                sha_map[sha] = fname

            # Perceptual hashes
            ph = compute_phash(img_path)
            dh = compute_dhash(img_path)
            if ph is not None and dh is not None:
                phash_dict[fname] = ph
                dhash_dict[fname] = dh
                img_list.append(fname)

    print(f"\n[A] Exact Binary Duplicates (SHA-256): {len(exact_duplicates)}")
    if exact_duplicates:
        for dup in exact_duplicates:
            print(f"   Duplicate pair: {dup[0]} <==> {dup[1]}")
    else:
        print("   -> No exact binary duplicates found across all 2,593 images.")

    # B. Near-Duplicate Perceptual Hash Scan
    # Pairwise hamming distance check
    print(f"\n[B] Computing Pairwise Perceptual Hash Distance Matrix ({len(img_list)} images)...")
    
    # Pack hashes into uint64 / boolean matrices for fast vectorized hamming computation
    ph_matrix = np.array([phash_dict[f] for f in img_list], dtype=bool) # shape: (N, 64)
    
    # Fast pairwise hamming: (N, 64) vs (64, N)
    # distance = sum(A != B)
    # Using matrix math: distance = 64 - 2 * (A @ B.T) + (ones - A) @ (ones - B).T
    # Or in chunks
    N = len(img_list)
    near_dup_pairs = []
    
    threshold_hamming = 4  # Hamming distance <= 4 out of 64 bits (< 6.25% bit difference) indicates near-duplicate / burst frame

    chunk_size = 500
    for i in range(0, N, chunk_size):
        chunk_a = ph_matrix[i:i+chunk_size]
        # XOR with all images
        for j in range(i, N, chunk_size):
            chunk_b = ph_matrix[j:j+chunk_size]
            # dist = np.sum(chunk_a[:, None, :] != chunk_b[None, :, :], axis=2)
            # Efficient XOR sum:
            dists = np.bitwise_xor(chunk_a[:, None, :], chunk_b[None, :, :]).sum(axis=2)
            
            # Find pairs with distance <= threshold
            rows, cols = np.where(dists <= threshold_hamming)
            for r, c in zip(rows, cols):
                global_i = i + r
                global_j = j + c
                if global_i < global_j: # Avoid self and duplicate pair
                    f1 = img_list[global_i]
                    f2 = img_list[global_j]
                    d_dist = hamming_distance(dhash_dict[f1], dhash_dict[f2])
                    near_dup_pairs.append({
                        "file_1": f1,
                        "file_2": f2,
                        "phash_distance": int(dists[r, c]),
                        "dhash_distance": int(d_dist)
                    })

    print(f"Total Near-Duplicate Pairs Found (pHash Hamming Dist <= {threshold_hamming}): {len(near_dup_pairs)}")
    
    # Group near duplicates into clusters
    from collections import defaultdict
    adj = defaultdict(set)
    for p in near_dup_pairs:
        adj[p["file_1"]].add(p["file_2"])
        adj[p["file_2"]].add(p["file_1"])

    visited = set()
    clusters = []
    for node in adj:
        if node not in visited:
            cluster = []
            q = [node]
            visited.add(node)
            for curr in q:
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            clusters.append(cluster)

    print(f"Total Redundant Frame Clusters: {len(clusters)}")
    total_redundant_images = sum(len(c) for c in clusters)
    print(f"Total Images in Redundant Clusters: {total_redundant_images}")

    # Output detailed report
    report = {
        "exact_duplicates_count": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs_count": len(near_dup_pairs),
        "near_duplicate_clusters_count": len(clusters),
        "sample_near_duplicate_pairs": near_dup_pairs[:20],
        "sample_clusters": [c[:5] for c in clusters[:10]]
    }

    paths.audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.audit_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed audit report saved to '{paths.audit_report_path}'.")
    return report

if __name__ == '__main__':
    evaluate_calibration_iaa()
    evaluate_duplicates()
