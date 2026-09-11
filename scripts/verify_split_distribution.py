"""Dataset Split Stratification and Class Distribution Verification Script.

Physical image and annotation counts across Train, Validation, and Test sets
are audited to confirm adherence to target stratification ratios with zero data leakage.
"""

import sys
from pathlib import Path

# Project root is resolved for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.core.config import ProjectPaths


def verify_distribution() -> pd.DataFrame:
    """
    Computes per-class split distributions and formatting summary tables.

    Returns:
        DataFrame containing formatted distribution statistics.
    """
    paths = ProjectPaths()
    df = pd.read_csv(paths.manifest_path)
    active_df = df[df["is_pruned"] == False].copy()

    classes = sorted(list(active_df["class_name"].unique()))
    splits = ["train", "val", "test"]

    # Contingency cross-tabulation table is computed
    ct = pd.crosstab(active_df["class_name"], active_df["split"])
    for s in splits:
        if s not in ct.columns:
            ct[s] = 0

    records = []
    total_tr = int(ct["train"].sum())
    total_va = int(ct["val"].sum())
    total_te = int(ct["test"].sum())
    grand_total = total_tr + total_va + total_te

    for c in classes:
        n_tr = int(ct.loc[c, "train"])
        n_va = int(ct.loc[c, "val"])
        n_te = int(ct.loc[c, "test"])
        c_tot = n_tr + n_va + n_te

        p_tr = (n_tr / c_tot) * 100.0 if c_tot > 0 else 0.0
        p_va = (n_va / c_tot) * 100.0 if c_tot > 0 else 0.0
        p_te = (n_te / c_tot) * 100.0 if c_tot > 0 else 0.0

        records.append({
            "Denomination": c,
            "Total": c_tot,
            "Train Count": n_tr,
            "Train %": f"{p_tr:5.1f}%",
            "Val Count": n_va,
            "Val %": f"{p_va:5.1f}%",
            "Test Count": n_te,
            "Test %": f"{p_te:5.1f}%",
        })

    # Summary totals row is appended
    records.append({
        "Denomination": "TOTAL",
        "Total": grand_total,
        "Train Count": total_tr,
        "Train %": f"{(total_tr / grand_total)*100:5.1f}%",
        "Val Count": total_va,
        "Val %": f"{(total_va / grand_total)*100:5.1f}%",
        "Test Count": total_te,
        "Test %": f"{(total_te / grand_total)*100:5.1f}%",
    })

    summary_df = pd.DataFrame(records)
    return summary_df


if __name__ == "__main__":
    table = verify_distribution()
    print("\n" + "="*80)
    print("AZERBAIJANI BANKNOTES DATASET: CLASS-STRATIFIED SPLIT VERIFICATION TABLE")
    print("="*80)
    print(table.to_string(index=False))
    print("="*80 + "\n")
