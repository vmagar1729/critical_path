import os
from src.critical_path.psplib.psplib_loader import sm_to_cpm_df
from src.critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df
from src.critical_path.psplib.loader import load_j30_optima
from src.critical_path.psplib.loader import parse_instance_number

def validate_folder(sm_folder, opt_path="j30opt.sm"):
    opt = load_j30_optima(opt_path)

    for filename in sorted(os.listdir(sm_folder)):
        if not filename.endswith(".sm"):
            continue

        inst_num = parse_instance_number(filename)
        optimal = opt.get(inst_num)

        df = sm_to_cpm_df(os.path.join(sm_folder, filename))

        res_df, bl_cp, lv_cp = compute_dual_cpm_from_df(df)
        proj_len = res_df["BL_EF"].max()

        status = "MATCH" if proj_len == optimal else f"OFF by {proj_len - optimal}"

        print(f"{filename:12s}   Inst={inst_num:3d}   "
              f"ProjLen={proj_len:5.1f}   Opt={optimal:3d}   {status}")