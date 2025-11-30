# critical_path/cpm/analytics_engine.py

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any


REQUIRED_ANALYTICS_FIELDS = [
    "BL_EF",
    "LV_EF",
    "PercentComplete",
    "Dur_BL",
    "Duration",
    "Float_LV",
]


def ensure_analytics_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all analytics-related derived fields exist.

    This is a thin layer on top of compute_dual_cpm_from_df output.
    It is defensive: if some fields are missing, it computes them
    when possible instead of crashing the dashboard.

    Adds / ensures:
      - ExpectedPct         (0–100)
      - ScheduleVariance    (PercentComplete - ExpectedPct)
      - DurationCreep       (Duration - Dur_BL)
      - HasDurationCreep    (bool)
      - Remaining_LV        (live remaining duration)
      - SlippageExposure    (Remaining_LV * CriticalityWeight)
      - CriticalityWeight   (fallback if missing)
    """

    df = df.copy()

    # Coerce some core numeric columns
    for col in ["PercentComplete", "Dur_BL", "Duration", "Float_LV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 1. ExpectedPct / ScheduleVariance
    # ------------------------------------------------------------------
    if "ExpectedPct" not in df.columns:
        if "Baseline Start" in df.columns and "Dur_BL" in df.columns:
            today = pd.Timestamp.today().normalize()
            dur = df["Dur_BL"].replace(0, np.nan)
            elapsed = (today - df["Baseline Start"]).dt.days.clip(lower=0)
            expected = (elapsed / dur).clip(lower=0, upper=1).fillna(0) * 100
            df["ExpectedPct"] = expected
        else:
            df["ExpectedPct"] = 0.0

    if "ScheduleVariance" not in df.columns:
        # PercentComplete is assumed to be 0–100
        df["ScheduleVariance"] = df["PercentComplete"].fillna(0) - df["ExpectedPct"].fillna(0)

    # ------------------------------------------------------------------
    # 2. Duration Creep
    # ------------------------------------------------------------------
    if "DurationCreep" not in df.columns or "HasDurationCreep" not in df.columns:
        if "Dur_BL" in df.columns and "Duration" in df.columns:
            creep = df["Duration"].fillna(df["Dur_BL"]) - df["Dur_BL"]
            df["DurationCreep"] = creep
            df["HasDurationCreep"] = creep > 0
        else:
            df["DurationCreep"] = 0.0
            df["HasDurationCreep"] = False

    # ------------------------------------------------------------------
    # 3. Remaining_LV
    # ------------------------------------------------------------------
    if "Remaining_LV" not in df.columns:
        # Use Dur_LV if present, else Duration, else Dur_BL
        if "Dur_LV" in df.columns:
            live_dur = df["Dur_LV"]
        elif "Duration" in df.columns:
            live_dur = df["Duration"]
        else:
            live_dur = df["Dur_BL"]

        live_dur = pd.to_numeric(live_dur, errors="coerce").fillna(0)
        pct = df["PercentComplete"].fillna(0) / 100.0
        df["Remaining_LV"] = (1 - pct).clip(lower=0) * live_dur

    # ------------------------------------------------------------------
    # 4. CriticalityWeight
    # ------------------------------------------------------------------
    if "CriticalityWeight" not in df.columns:
        # Fallback heuristic: 1.0 for critical, 0.6 for near-zero float, else 0.1
        def _crit_weight(row):
            fl = row.get("Float_LV", np.nan)
            if pd.isna(fl):
                return 0.1
            if abs(fl) < 1e-6:
                return 1.0
            if 0 < fl <= 1.0:
                return 0.6
            return 0.1

        df["CriticalityWeight"] = df.apply(_crit_weight, axis=1)

    # ------------------------------------------------------------------
    # 5. SlippageExposure
    # ------------------------------------------------------------------
    if "SlippageExposure" not in df.columns:
        df["SlippageExposure"] = df["Remaining_LV"].fillna(0) * df["CriticalityWeight"].fillna(0)

    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute high-level analytics KPIs for the dashboard.

    Returns a dict with:
      - total_tasks
      - leaf_tasks
      - critical_tasks
      - behind_tasks
      - avg_percent_complete
      - total_remaining
      - total_slippage_exposure
    """

    out: Dict[str, Any] = {}

    out["total_tasks"] = int(len(df))

    if "IsLeaf" in df.columns:
        out["leaf_tasks"] = int(df["IsLeaf"].sum())
    else:
        out["leaf_tasks"] = int(len(df))

    if "Float_LV" in df.columns:
        out["critical_tasks"] = int((df["Float_LV"].abs() < 1e-6).sum())
    else:
        out["critical_tasks"] = 0

    if "ScheduleVariance" in df.columns:
        out["behind_tasks"] = int((df["ScheduleVariance"] < 0).sum())
    else:
        out["behind_tasks"] = 0

    out["avg_percent_complete"] = float(df["PercentComplete"].fillna(0).mean())

    if "Remaining_LV" in df.columns:
        out["total_remaining"] = float(df["Remaining_LV"].fillna(0).sum())
    else:
        out["total_remaining"] = 0.0

    if "SlippageExposure" in df.columns:
        out["total_slippage_exposure"] = float(df["SlippageExposure"].fillna(0).sum())
    else:
        out["total_slippage_exposure"] = 0.0

    return out


def add_float_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'FloatBucket' column grouping live float into friendly buckets.
    """

    df = df.copy()

    if "Float_LV" not in df.columns:
        df["FloatBucket"] = "Unknown"
        return df

    f = df["Float_LV"].fillna(0)

    bins = [-1e9, -0.01, 0.01, 1, 5, 10, 999999999]
    labels = [
        "Negative float",
        "Critical (0)",
        "≤ 1 day",
        "1–5 days",
        "5–10 days",
        "> 10 days",
    ]
    df["FloatBucket"] = pd.cut(f, bins=bins, labels=labels, include_lowest=True)
    df["FloatBucket"] = df["FloatBucket"].astype(str)

    return df