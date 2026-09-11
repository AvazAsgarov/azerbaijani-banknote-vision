"""
Legacy Raw Data Consolidation Script.

Active dataset curation and zero-leakage splitting pipelines reside within `src/data/`.
This module is preserved for historical raw ingestion provenance.
"""

import os
import sys
import shutil
import zipfile
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths

def main() -> None:
    """Entrypoint for legacy raw data consolidation and ingestion provenance."""
    paths = ProjectPaths()
    if "--force" not in sys.argv:
        print(f"Note: Active standardized dataset resides in '{paths.processed_dir}'.")
        print("To rerun historical raw extraction, pass '--force'. Exiting safely.")
        return
    print("=== 1. Loading All Assignment Metadata ===")
    dfs = []
    for member in ['avaz', 'gulnar', 'hasan', 'kazim']:
        path = paths.root_dir / f'AzeBankNotes_labelling/{member}/assignment.csv'
        df = pd.read_csv(path)
        df['annotated_by'] = member
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records in assignment: {len(all_df)}")

    base_dir = paths.root_dir / 'dataset'
    if base_dir.exists():
        shutil.rmtree(base_dir)

    for split in ['train', 'val', 'test']:
        os.makedirs(base_dir / 'images' / split, exist_ok=True)
        os.makedirs(base_dir / 'labels' / split, exist_ok=True)

    os.makedirs(os.path.join(base_dir, 'master', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'master', 'labels'), exist_ok=True)

    print("\n=== 2. Computing Zero-Leakage Stratified Group Split ===")
    classes = ['001_azn', '005_azn', '010_azn', '020_azn', '050_azn', '100_azn', '200_azn']
    folder_matrix = pd.crosstab(all_df['original_capture_folder'], all_df['class_name'])
    total_per_class = all_df['class_name'].value_counts()
    folders = list(folder_matrix.index)

    best_score = float('inf')
    best_split = None

    for seed in range(5000):
        np.random.seed(seed)
        shuffled_folders = np.random.permutation(folders)
        
        train_f, val_f, test_f = [], [], []
        train_counts = np.zeros(len(classes))
        val_counts = np.zeros(len(classes))
        test_counts = np.zeros(len(classes))
        
        for f in shuffled_folders:
            f_counts = folder_matrix.loc[f].values
            train_ratio = train_counts / (total_per_class.values * 0.70 + 1e-6)
            val_ratio = val_counts / (total_per_class.values * 0.15 + 1e-6)
            test_ratio = test_counts / (total_per_class.values * 0.15 + 1e-6)
            
            scores = [
                np.mean(train_ratio),
                np.mean(val_ratio),
                np.mean(test_ratio)
            ]
            chosen = np.argmin(scores)
            if chosen == 0:
                train_f.append(f)
                train_counts += f_counts
            elif chosen == 1:
                val_f.append(f)
                val_counts += f_counts
            else:
                test_f.append(f)
                test_counts += f_counts
                
        n_train = np.sum(train_counts)
        n_val = np.sum(val_counts)
        n_test = np.sum(test_counts)
        total = n_train + n_val + n_test
        
        prop_err = abs(n_train/total - 0.70) + abs(n_val/total - 0.15) + abs(n_test/total - 0.15)
        class_err = sum(abs(train_counts[i]/total_per_class[c] - 0.70) + 
                        abs(val_counts[i]/total_per_class[c] - 0.15) + 
                        abs(test_counts[i]/total_per_class[c] - 0.15) 
                        for i, c in enumerate(classes))
        
        score = prop_err * 3.0 + class_err
        if score < best_score:
            best_score = score
            best_split = (set(train_f), set(val_f), set(test_f), train_counts, val_counts, test_counts, seed)

    train_f, val_f, test_f, train_counts, val_counts, test_counts, seed = best_split
    print(f"Optimal Split Found (Seed {seed})!")
    print(f"Train Folders: {len(train_f)} | Val Folders: {len(val_f)} | Test Folders: {len(test_f)}")
    print(f"Train Images: {int(np.sum(train_counts))} ({np.sum(train_counts)/2593*100:.1f}%)")
    print(f"Val Images:   {int(np.sum(val_counts))} ({np.sum(val_counts)/2593*100:.1f}%)")
    print(f"Test Images:  {int(np.sum(test_counts))} ({np.sum(test_counts)/2593*100:.1f}%)")

    folder_to_split = {}
    for f in train_f: folder_to_split[f] = 'train'
    for f in val_f: folder_to_split[f] = 'val'
    for f in test_f: folder_to_split[f] = 'test'

    all_df['split'] = all_df['original_capture_folder'].map(folder_to_split)

    print("\n=== 3. Extracting and Consolidating Files ===")
    g_zip = zipfile.ZipFile('gulnar images.zip')
    g_img_map = {re.sub(r'^[0-9a-fA-F]{8}-', '', os.path.basename(f)): f for f in g_zip.namelist() if f.startswith('images/') and f.endswith('.jpg')}
    g_lbl_map = {re.sub(r'^[0-9a-fA-F]{8}-', '', os.path.basename(f)): f for f in g_zip.namelist() if f.startswith('labels/') and f.endswith('.txt')}

    h_zip = zipfile.ZipFile('hasan')
    h_img_map = {os.path.basename(f): f for f in h_zip.namelist() if f.startswith('labels/') and f.endswith('.jpg')}
    h_lbl_map = {os.path.basename(f): f for f in h_zip.namelist() if f.startswith('labels/') and f.endswith('.txt')}

    k_zip = zipfile.ZipFile('kazim_labeled.zip')
    k_img_map = {os.path.basename(f): f for f in k_zip.namelist() if f.startswith('kazim/images/') and f.endswith('.jpg')}
    k_lbl_map = {os.path.basename(f): f for f in k_zip.namelist() if f.startswith('kazim/images_label/') and f.endswith('.txt')}

    box_counts_per_split = {'train': 0, 'val': 0, 'test': 0}
    box_counts_per_class = {c: 0 for c in classes}
    images_processed = 0

    # Also track bounding box count per image
    bbox_counts = []

    for idx, row in all_df.iterrows():
        fname = row['filename']
        lbl_name = fname.replace('.jpg', '.txt').replace('.png', '.txt')
        annotator = row['annotated_by']
        split = row['split']
        
        dest_img = os.path.join(base_dir, 'images', split, fname)
        dest_lbl = os.path.join(base_dir, 'labels', split, lbl_name)
        master_img = os.path.join(base_dir, 'master', 'images', fname)
        master_lbl = os.path.join(base_dir, 'master', 'labels', lbl_name)
        
        img_bytes = None
        lbl_text = ''
        
        if annotator == 'avaz':
            src_img = os.path.join('AzeBankNotes_labelling/avaz/images', fname)
            src_lbl = os.path.join('AzeBankNotes_labelling/avaz/labels', lbl_name)
            if os.path.exists(src_img):
                with open(src_img, 'rb') as f: img_bytes = f.read()
            if os.path.exists(src_lbl):
                with open(src_lbl, 'r', encoding='utf-8') as f: lbl_text = f.read()
        elif annotator == 'gulnar':
            if fname in g_img_map:
                img_bytes = g_zip.read(g_img_map[fname])
            elif os.path.exists(f'AzeBankNotes_labelling/gulnar/images/{fname}'):
                with open(f'AzeBankNotes_labelling/gulnar/images/{fname}', 'rb') as f: img_bytes = f.read()
            if lbl_name in g_lbl_map:
                lbl_text = g_zip.read(g_lbl_map[lbl_name]).decode('utf-8')
        elif annotator == 'hasan':
            if fname in h_img_map:
                img_bytes = h_zip.read(h_img_map[fname])
            if lbl_name in h_lbl_map:
                lbl_text = h_zip.read(h_lbl_map[lbl_name]).decode('utf-8')
        elif annotator == 'kazim':
            if fname in k_img_map:
                img_bytes = k_zip.read(k_img_map[fname])
            elif os.path.exists(f'AzeBankNotes_labelling/kazim/images/{fname}'):
                with open(f'AzeBankNotes_labelling/kazim/images/{fname}', 'rb') as f: img_bytes = f.read()
            if lbl_name in k_lbl_map:
                lbl_text = k_zip.read(k_lbl_map[lbl_name]).decode('utf-8')
                
        if img_bytes is not None:
            with open(dest_img, 'wb') as f: f.write(img_bytes)
            with open(master_img, 'wb') as f: f.write(img_bytes)
            
            clean_lines = []
            for l in lbl_text.strip().splitlines():
                l_str = l.strip()
                if l_str:
                    parts = l_str.split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        clean_lines.append(l_str)
                        box_counts_per_split[split] += 1
                        box_counts_per_class[classes[cls_id]] += 1
                        
            cleaned_lbl_content = '\n'.join(clean_lines) + ('\n' if clean_lines else '')
            with open(dest_lbl, 'w', encoding='utf-8') as f: f.write(cleaned_lbl_content)
            with open(master_lbl, 'w', encoding='utf-8') as f: f.write(cleaned_lbl_content)
            images_processed += 1
            bbox_counts.append(len(clean_lines))

    all_df['num_boxes'] = bbox_counts

    # Save dataset_manifest.csv
    manifest_path = os.path.join(base_dir, 'dataset_manifest.csv')
    all_df.to_csv(manifest_path, index=False)
    print(f"Saved dataset_manifest.csv with {len(all_df)} rows.")

    # Save data.yaml
    yaml_lines = [
        "# Ultralytics YOLOv8 / YOLOv11 / RT-DETR Dataset Configuration",
        "# Azerbaijani Banknotes (AZN) Detection Dataset",
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: 001_azn",
        "  1: 005_azn",
        "  2: 010_azn",
        "  3: 020_azn",
        "  4: 050_azn",
        "  5: 100_azn",
        "  6: 200_azn",
        ""
    ]
    with open(os.path.join(base_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(yaml_lines))
    print("Saved data.yaml.")

    # Save dataset_summary.json
    summary = {
        "dataset_name": "Azerbaijani Banknotes (AZN) Detection Dataset",
        "total_images": images_processed,
        "total_bounding_boxes": sum(box_counts_per_split.values()),
        "split_image_counts": {
            "train": len(all_df[all_df['split'] == 'train']),
            "val": len(all_df[all_df['split'] == 'val']),
            "test": len(all_df[all_df['split'] == 'test'])
        },
        "split_box_counts": box_counts_per_split,
        "class_box_counts": box_counts_per_class,
        "annotator_contributions": all_df['annotated_by'].value_counts().to_dict(),
        "total_capture_environments": len(folders),
        "leakage_isolation": "100% Zero-Leakage (Disjoint capture scenes between Train, Val, Test)"
    }
    with open(os.path.join(base_dir, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("Saved dataset_summary.json.")

    print("\n=== 4. Final Directory Verification ===")
    for s in ['train', 'val', 'test']:
        n_i = len(os.listdir(os.path.join(base_dir, 'images', s)))
        n_l = len(os.listdir(os.path.join(base_dir, 'labels', s)))
        print(f"Split [{s}]: {n_i} images, {n_l} labels (1:1 Paired: {n_i == n_l})")
    n_master_i = len(os.listdir(os.path.join(base_dir, 'master', 'images')))
    n_master_l = len(os.listdir(os.path.join(base_dir, 'master', 'labels')))
    print(f"Master: {n_master_i} images, {n_master_l} labels")

if __name__ == '__main__':
    main()
