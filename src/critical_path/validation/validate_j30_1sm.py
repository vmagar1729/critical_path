import sys
from pathlib import Path

import pandas as pd

from src.critical_path.psplib.psplib_loader import sm_to_cpm_df
from src.critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df


def check_critical_path_monotonic(df: pd.DataFrame, cp_nodes):
    """
    Basic sanity checks on the CP:
    - nodes strictly follow BL_ES order
    - BL_EF is non-decreasing along the path
    """
    cp_df = df[df["TaskID"].isin(cp_nodes)].copy()

    # Ensure we kept ordering consistent with ES (ties allowed)
    cp_df = cp_df.sort_values("BL_ES")

    es_values = cp_df["BL_ES"].tolist()
    ef_values = cp_df["BL_EF"].tolist()

    es_ok = all(es_values[i] <= es_values[i + 1] for i in range(len(es_values) - 1))
    ef_ok = all(ef_values[i] <= ef_values[i + 1] for i in range(len(ef_values) - 1))

    return es_ok and ef_ok


def validate_instance(sm_path: Path):
    """Run CPM on a single .sm file and print a compact summary."""
    try:
        df_raw = sm_to_cpm_df(str(sm_path))
        res_df, bl_cp, lv_cp = compute_dual_cpm_from_df(df_raw)
    except Exception as e:
        print(f"[{sm_path.name}] ERROR during CPM: {e}")
        return

    # Project baseline length
    proj_len = float(res_df["BL_EF"].max())

    # Number of nodes on baseline CP
    cp_len = len(bl_cp) if bl_cp is not None else 0

    # Internal sanity checks
    issues = []

    # 1. No NaNs in Dur_BL
    if res_df["Dur_BL"].isna().any():
        issues.append("NaN Dur_BL")

    # 2. CP monotonic in time
    if bl_cp:
        if not check_critical_path_monotonic(res_df, bl_cp):
            issues.append("non-monotonic CP")

    status = "OK" if not issues else "WARN: " + ", ".join(issues)

    print(
        f"{sm_path.name:15s}  "
        f"ProjLen={proj_len:5.1f}  "
        f"CP_nodes={cp_len:2d}  "
        f"{status}"
    )


def main():
    if len(sys.argv) > 1:
        j30_dir = Path(sys.argv[1])
    else:
        # default relative path: ../j30 from critical_path/
        j30_dir = Path(__file__).resolve().parent.parent / "j30"

    if not j30_dir.exists():
        print(f"j30 directory not found: {j30_dir}")
        sys.exit(1)

    sm_files = sorted(j30_dir.glob("*.sm"))
    if not sm_files:
        print(f"No .sm files found in {j30_dir}")
        sys.exit(1)

    print(f"Running validator over {len(sm_files)} instances in {j30_dir}…\n")

    for sm in sm_files:
        validate_instance(sm)


if __name__ == "__main__":
    main()