import os
import pandas as pd
from src.critical_path.psplib.psplib_loader import sm_to_cpm_df
from src.critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df

J30_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "j30")
J30_DIR = os.path.abspath(J30_DIR)

def validate_all_j30():
    results = []

    sm_files = [f for f in os.listdir(J30_DIR) if f.lower().endswith(".sm")]
    sm_files.sort()

    print(f"Found {len(sm_files)} J30 instances.")
    print("Starting validation...\n")

    for fname in sm_files:
        path = os.path.join(J30_DIR, fname)
        print(f"Processing {fname} ...", end=" ")

        try:
            df = sm_to_cpm_df(path)
            df_res, bl_cp, lv_cp = compute_dual_cpm_from_df(df)

            project_duration = df_res["BL_EF"].max()

            results.append({
                "file": fname,
                "duration": project_duration,
                "critical_path": bl_cp,
                "status": "OK"
            })

            print("OK")

        except Exception as e:
            results.append({
                "file": fname,
                "duration": None,
                "critical_path": None,
                "status": f"ERROR: {str(e)}"
            })

            print(f"ERROR ({e})")

    # Save summary
    results_df = pd.DataFrame(results)
    out_path = os.path.join(J30_DIR, "batch_results.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nBatch complete. Results saved to: {out_path}")
    return results_df


if __name__ == "__main__":
    validate_all_j30()