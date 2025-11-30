import sys
import os

# -----------------------------------------------------------
# Make sure "critical_path" package is importable
# -----------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
PROJECT_SRC = os.path.abspath(os.path.join(THIS_FILE, "../../../"))

if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df

# -----------------------------------------------------------
# Page config (only if this is the main script)
# -----------------------------------------------------------
st.set_page_config(
    page_title="Executive Project Summary",
    layout="wide",
)

st.title("🧭 Executive Project Summary")

# st.markdown(
#     """
# This page answers three questions your sponsors actually care about:
#
# 1. **Are we on track?**
# 2. **If not, how far behind and who owns the delay?**
# 3. **Can we realistically make it up with current float?**
# """
# )

# -----------------------------------------------------------
# Upload
# -----------------------------------------------------------
# uploaded = st.file_uploader("Upload Schedule CSV", type=["csv"])
#
# if uploaded is None:
#     st.info("Upload a CSV exported from MS Project or your generator.")
#     st.stop()
#
# try:
#     df_raw = pd.read_csv(uploaded)
# except Exception as e:
#     st.error(f"Error parsing CSV: {e}")
#     st.stop()

# Use the central session schedule
df_raw = st.session_state.get("schedule_df", None)

# Fix datetime fields
date_cols = ["Start", "Finish", "Baseline Start", "Baseline Finish"]
for c in date_cols:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

# Owner normalization
if "Owner" not in df_raw.columns:
    df_raw["Owner"] = "Unassigned"
df_raw["Owner"] = df_raw["Owner"].fillna("Unassigned").astype(str)

# -----------------------------------------------------------
# Run CPM Engine
# -----------------------------------------------------------
try:
    df = compute_dual_cpm_from_df(df_raw)
except Exception as e:
    st.error(f"Error computing CPM: {e}")
    st.stop()

# Safety: coerce key numeric fields
for col in ["BL_EF", "LV_EF", "ScheduleVariance", "SlippageExposure",
            "Float_LV", "PercentComplete", "ExpectedPct",
            "Remaining_LV"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------------------------------------
# 1. Are we on track?
# -----------------------------------------------------------
bl_finish = float(df["BL_EF"].max()) if "BL_EF" in df.columns else np.nan
lv_finish = float(df["LV_EF"].max()) if "LV_EF" in df.columns else np.nan

if np.isfinite(bl_finish) and np.isfinite(lv_finish):
    slip_days = lv_finish - bl_finish
else:
    slip_days = 0.0

# Status classification
if slip_days <= 0.5:
    status_label = "On Track"
    status_color = "✅"
    status_desc = "Live finish is at or ahead of baseline."
elif slip_days <= 3.0:
    status_label = "At Risk"
    status_color = "🟡"
    status_desc = "We are slipping, but still within a manageable window."
else:
    status_label = "Behind"
    status_color = "🔴"
    status_desc = "Baseline date is in the rearview mirror. This will be noticed."

st.markdown("## 1. Overall Status")

c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    st.metric(
        "Baseline Finish (days from project start)",
        f"{bl_finish:.1f}" if np.isfinite(bl_finish) else "N/A",
    )

with c2:
    st.metric(
        "Live Predicted Finish",
        f"{lv_finish:.1f}" if np.isfinite(lv_finish) else "N/A",
        delta=f"{slip_days:+.1f} days",
    )

with c3:
    st.markdown(
        f"""
        ### {status_color} {status_label}  
        {status_desc}
        """
    )

st.divider()

# -----------------------------------------------------------
# 2. If we are behind – how much and who owns it?
# -----------------------------------------------------------
st.markdown("## 2. Where is the delay, and who owns it?")

if "ScheduleVariance" not in df.columns:
    st.error("ScheduleVariance not found in DF. Check intelligence layer wiring.")
    st.stop()

behind_mask = df["ScheduleVariance"] < 0
behind_df = df[behind_mask].copy()

if behind_df.empty:
    st.success("No tasks are behind schedule. No throats require attention today.")
else:
    # Default slippage exposure if missing
    if "SlippageExposure" not in behind_df.columns:
        behind_df["SlippageExposure"] = (
            behind_df["Remaining_LV"].fillna(0.0).abs()
        )

    # Aggregate by Owner
    owner_group = (
        behind_df
        .groupby("Owner", dropna=False)
        .agg(
            TasksBehind=("TaskID", "count"),
            TotalExposure=("SlippageExposure", "sum"),
            MaxSlip=("ScheduleVariance", lambda x: float(x.min()) if len(x) else np.nan),
        )
        .reset_index()
    )

    # Convert exposure to %
    total_exposure = owner_group["TotalExposure"].sum()
    if total_exposure > 0:
        owner_group["ExposurePct"] = owner_group["TotalExposure"] / total_exposure * 100
    else:
        owner_group["ExposurePct"] = 0.0

    # Sort for display
    owner_group = owner_group.sort_values("TotalExposure", ascending=False)

    col_pie, col_table = st.columns([2, 3])

    with col_pie:
        st.markdown("### Slippage Exposure by Owner")
        fig_owner = px.pie(
            owner_group,
            names="Owner",
            values="TotalExposure",
            hover_data=["TasksBehind", "MaxSlip"],
            hole=0.4,
        )
        fig_owner.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_owner, use_container_width=True)

    with col_table:
        st.markdown("### Owner Impact Table")
        pretty_owner = owner_group.copy()
        pretty_owner["TotalExposure"] = pretty_owner["TotalExposure"].round(1)
        pretty_owner["MaxSlip"] = pretty_owner["MaxSlip"].round(1)
        pretty_owner["ExposurePct"] = pretty_owner["ExposurePct"].round(1)
        st.dataframe(
            pretty_owner[
                ["Owner", "TasksBehind", "TotalExposure", "ExposurePct", "MaxSlip"]
            ]
        )

    # Top delayed tasks
    st.markdown("### Top Delayed Tasks")

    top_tasks = (
        behind_df
        .copy()
        .assign(SlipDays=lambda d: d["ScheduleVariance"] / 100.0 * d["Dur_BL"])
    )

    # Fallback if Dur_BL missing or zero
    top_tasks["SlipDays"] = top_tasks["SlipDays"].fillna(0.0)
    top_tasks = top_tasks.sort_values("ScheduleVariance")  # most negative first
    top_tasks = top_tasks.head(15)

    fig_tasks = px.bar(
        top_tasks,
        x="SlipDays",
        y="Name",
        color="Owner",
        orientation="h",
        hover_data=["TaskID", "PercentComplete", "ExpectedPct"],
        labels={"SlipDays": "Estimated Days Behind"},
    )
    fig_tasks.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_tasks, use_container_width=True)

st.divider()

# -----------------------------------------------------------
# 3. Can we make it up?
# -----------------------------------------------------------
st.markdown("## 3. Can we realistically recover?")

if "Float_LV" not in df.columns:
    st.error("Float_LV not found. Check CPM computation wiring.")
    st.stop()

# Recoverable = behind AND positive float
recoverable_mask = (df["ScheduleVariance"] < 0) & (df["Float_LV"] > 0)
total_recoverable = float(df.loc[recoverable_mask, "Float_LV"].sum())

if slip_days <= 0.5:
    recovery_label = "No recovery needed"
    recovery_msg = (
        "Live finish is aligned with baseline. Any micro-variances are noise."
    )
    recovery_color = "✅"
elif slip_days > 0 and total_recoverable >= slip_days:
    recovery_label = "Delay is recoverable using existing float"
    recovery_msg = (
        "There is enough slack in non-critical / near-critical tasks to absorb "
        "the current delay without moving the finish line, if you actually act on it."
    )
    recovery_color = "🟢"
elif slip_days > 0 and 0 < total_recoverable < slip_days:
    recovery_label = "Partially recoverable"
    recovery_msg = (
        f"We can claw back **~{total_recoverable:.1f} days**, but the remaining "
        f"**~{slip_days - total_recoverable:.1f} days** will require either "
        "re-sequencing or throwing people/money at the problem."
    )
    recovery_color = "🟠"
else:
    recovery_label = "Not recoverable with current plan"
    recovery_msg = (
        "Float on the network is effectively exhausted. Any meaningful recovery "
        "will need fast-tracking, scope trade-offs, or acceptance of a later go-live."
    )
    recovery_color = "🔴"

c_rec1, c_rec2 = st.columns([2, 3])

with c_rec1:
    st.metric("Current Slip (days)", f"{slip_days:.1f}")
    st.metric("Total Recoverable Float (days)", f"{total_recoverable:.1f}")

with c_rec2:
    st.markdown(
        f"""
        ### {recovery_color} {recovery_label}  
        {recovery_msg}
        """
    )
#
# st.divider()
#
# # -----------------------------------------------------------
# # Optional: Raw DF for nerds (i.e., you)
# # -----------------------------------------------------------
# with st.expander("Nerd view: full CPM dataframe"):
#     st.dataframe(df)