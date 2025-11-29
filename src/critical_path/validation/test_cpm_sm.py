from src.critical_path.psplib.psplib_loader import sm_to_cpm_df
from dual_cpm_csv import compute_dual_cpm_from_df

df = sm_to_cpm_df("/j30/j301_1.sm")
res_df, bl_cp, lv_cp = compute_dual_cpm_from_df(df)

cp = [1, 3, 8, 12, 14, 17, 22, 23, 24, 30, 32]
print(res_df.loc[res_df["TaskID"].isin(cp), ["TaskID", "Dur_BL", "BL_ES", "BL_EF"]])
print("BL_EF max:", res_df["BL_EF"].max())

print("Baseline CP:", bl_cp)
print("Project baseline length (days):", res_df["BL_EF"].max())