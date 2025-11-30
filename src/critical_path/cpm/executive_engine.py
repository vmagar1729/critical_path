import pandas as pd
import numpy as np


def compute_executive_metrics(df: pd.DataFrame) -> dict:
    """
    Computes the high-level metrics needed for the Executive Overview page.

    Returns a dict:
      {
        bl_finish: float,
        lv_finish: float,
        slip: float,
        status_label: str,
        status_color: str,
        status_desc: str,
        owner_exposure_df: DataFrame,
        behind_df: DataFrame,
        creep_df: DataFrame,
        total_recoverable: float
      }
    """

    # Coerce numerics
    for col in ["BL_EF", "LV_EF", "ScheduleVariance", "SlippageExposure",
                "Float_LV", "PercentComplete", "ExpectedPct", "Remaining_LV",
                "Duration", "Dur_BL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic finish metrics
    bl_finish = float(df["BL_EF"].max())
    lv_finish = float(df["LV_EF"].max())
    slip = lv_finish - bl_finish

    # Status classification
    if slip <= 0.5:
        status_label = "On Track"
        status_color = "✅"
        status_desc = "Live finish is at or ahead of baseline."
    elif slip <= 3.0:
        status_label = "At Risk"
        status_color = "🟡"
        status_desc = "We’re slipping, but still within a reasonable window."
    else:
        status_label = "Behind"
        status_color = "🔴"
        status_desc = "Baseline date is now wishful thinking."

    # Slippage / Behind tasks
    behind_df = df[df["ScheduleVariance"] < 0].copy()

    # Exposure by owner
    if "SlippageExposure" not in df.columns:
        df["SlippageExposure"] = df["Remaining_LV"].fillna(0.0)

    owner_group = (
        behind_df
        .groupby("Owner", dropna=False)
        .agg(
            TasksBehind=("TaskID", "count"),
            TotalExposure=("SlippageExposure", "sum"),
            MaxSlip=("ScheduleVariance", "min"),
        )
        .reset_index()
    )

    if not owner_group.empty:
        total_exp = owner_group["TotalExposure"].sum()
        owner_group["ExposurePct"] = owner_group["TotalExposure"] / max(total_exp, 1e-9) * 100
    else:
        owner_group["ExposurePct"] = 0.0

    # Duration creep
    if "Duration" in df.columns and "Dur_BL" in df.columns:
        creep_df = df[df["Duration"] > df["Dur_BL"]].copy()
        creep_df["DurationCreep"] = creep_df["Duration"] - creep_df["Dur_BL"]
    else:
        creep_df = df.head(0).copy()

    # Recoverable float
    recoverable_df = df[(df["ScheduleVariance"] < 0) & (df["Float_LV"] > 0)]
    total_recoverable = float(recoverable_df["Float_LV"].sum())

    return {
        "bl_finish": bl_finish,
        "lv_finish": lv_finish,
        "slip": slip,
        "status_label": status_label,
        "status_color": status_color,
        "status_desc": status_desc,
        "owner_exposure_df": owner_group,
        "behind_df": behind_df,
        "creep_df": creep_df,
        "total_recoverable": total_recoverable,
    }