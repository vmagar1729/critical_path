import os
import pandas as pd
from src.critical_path.psplib.psplib_loader import sm_to_cpm_df, load_j30_optimal_durations
from src.critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df

# Detect directory holding the j30 files
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
J30_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "j30"))
OPT_FILE = os.path.join(J30_DIR, "j30opt.sm")

def validate_all_j30_with_opt():
    # Load optimal durations
    if not os.path.exists(OPT_FILE):
        raise FileNotFoundError(f"Optimal duration file not found: {OPT_FILE}")

    optimal = load_j30_optimal_durations(OPT_FILE)

    results = []

    sm_files = [f for f in os.listdir(J30_DIR) if f.lower().endswith(".sm")]
    sm_files.sort()

    print(f"Found {len(sm_files)} J30 instances.")
    print("Starting validation with optimal comparison...\n")

    for fname in sm_files:
        path = os.path.join(J30_DIR, fname)
        print(f"Processing {fname} ...", end=" ")

        opt = optimal.get(fname, None)

        try:
            df = sm_to_cpm_df(path)
            df_res, bl_cp, lv_cp = compute_dual_cpm_from_df(df)

            actual = df_res["BL_EF"].max()

            variance = None
            if opt is not None and actual is not None:
                variance = actual - opt

            results.append({
                "file": fname,
                "optimal_duration": opt,
                "actual_duration": actual,
                "variance": variance,
                "critical_path": bl_cp,
                "status": "OK"
            })

            print("OK")

        except Exception as e:
            results.append({
                "file": fname,
                "optimal_duration": opt,
                "actual_duration": None,
                "variance": None,
                "critical_path": None,
                "status": f"ERROR: {str(e)}"
            })
            print(f"ERROR ({e})")

    # Save summary
    results_df = pd.DataFrame(results)
    out_path = os.path.join(J30_DIR, "batch_results_with_opt.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nBatch complete. Results saved to: {out_path}")
    return results_df


if __name__ == "__main__":
    validate_all_j30_with_opt()