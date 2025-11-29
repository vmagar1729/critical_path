import sys
import pandas as pd
from pathlib import Path

from psplib_loader import sm_to_cpm_df
from dual_cpm_csv import compute_dual_cpm_from_df


def test_single_sm(sm_path):
    sm_path = Path(sm_path)
    print(f"\n=== Testing {sm_path.name} ===")

    df = sm_to_cpm_df(sm_path)
    print(f"Loaded {len(df)} tasks from {sm_path.name}")

    # Run your CPM engine
    res_df, bl_cp, lv_cp = compute_dual_cpm_from_df(df)

    # Use baseline CPM only for now (everything is 0% complete)
    leaf = res_df[res_df["IsLeaf"]]

    # Project duration from baseline early finish
    if not leaf["BL_EF"].dropna().empty:
        proj_len = leaf["BL_EF"].max()
    else:
        proj_len = None

    print(f"Baseline critical path (TaskIDs): {bl_cp}")
    print(f"Baseline project length (days):  {proj_len}")
    print(f"Number of leaf tasks:             {len(leaf)}")

    # Sanity check: all BL_Float >= 0
    bad_float = leaf[leaf["BL_Float"] < 0]
    if not bad_float.empty:
        print("⚠ Negative float detected in:")
        print(bad_float[["TaskID", "BL_Float"]])
    else:
        print("✔ No negative float.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_psplib_cpm.py path/to/j301_1.sm")
        sys.exit(1)

    test_single_sm(sys.argv[1])