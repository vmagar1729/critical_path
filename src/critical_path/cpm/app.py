import sys
import os

# -----------------------------------------------------------
# Path setup so "critical_path" package can be imported
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
# Visual constants – Bold Flat Palette (B)
# -----------------------------------------------------------
FLAT_COLORS = {
    "blue": "#1E88E5",   # primary
    "green": "#43A047",  # success
    "amber": "#FB8C00",  # warning
    "red": "#E53935",    # danger
    "purple": "#8E24AA", # accent
    "grey": "#757575",   # neutral
}

COLOR_SEQUENCE = [
    FLAT_COLORS["blue"],
    FLAT_COLORS["green"],
    FLAT_COLORS["amber"],
    FLAT_COLORS["red"],
    FLAT_COLORS["purple"],
]

# -----------------------------------------------------------
# Streamlit page config – MUST be first Streamlit call
# -----------------------------------------------------------
st.set_page_config(
    page_title="Executive CPM Overview",
    layout="wide",
)

st.title("📊 Executive CPM Overview")

st.caption(
    "Baseline vs Live schedule, critical paths, and risk exposure – "
    "all the things people pretend to understand in steering meetings."
)

# -----------------------------------------------------------
# File upload
# -----------------------------------------------------------
uploaded = st.file_uploader("Upload Schedule CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV exported from MS Project or your synthetic generator.")
    st.stop()

# Load CSV
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error parsing CSV: {e}")
    st.stop()

# Parse dates as timestamps
date_cols = ["Start", "Finish", "Baseline Start", "Baseline Finish"]
for c in date_cols:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

# -----------------------------------------------------------
# Run CPM engine
# -----------------------------------------------------------
try:
    df = compute_dual_cpm_from_df(df_raw)
except Exception as e:
    st.error(f"Error computing CPM: {e}")
    st.stop()

# Ensure required fields exist, otherwise abort gracefully
required_cols = [
    "TaskID", "Name", "WBS", "Outline Level",
    "BL_ES", "BL_EF", "LV_ES", "LV_EF",
    "Dur_BL", "Duration",
    "PercentComplete",
    "IsCritical_BL", "IsCritical_LV",
    "IsNearCritical_LV",
    "BecameCritical",
    "ScheduleVariance",
    "SlippageExposure",
    "Remaining_LV",
    "CriticalityWeight",
    "IsSummary", "IsLeaf"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Engine output missing expected columns: {missing}")
    st.stop()

# -----------------------------------------------------------
# High-level metrics
# -----------------------------------------------------------
bl_finish = float(df["BL_EF"].max()) if not df["BL_EF"].isna().all() else 0.0
lv_finish = float(df["LV_EF"].max()) if not df["LV_EF"].isna().all() else bl_finish

slip_days = float(lv_finish - bl_finish)

crit_live = int(df["IsCritical_LV"].sum())
near_crit_live = int(df["IsNearCritical_LV"].sum())
new_crit = int(df["BecameCritical"].sum())

behind = df[df["ScheduleVariance"] < 0]
behind_count = len(behind)

total_slip_exposure = float(df["SlippageExposure"].sum())

# Phase extraction (top-level WBS like 1, 2, 3)
df["PhaseKey"] = df["WBS"].astype(str).str.split(".").str[0]

phase_names = (
    df[df["Outline Level"] == 1]
    .dropna(subset=["WBS"])
    .drop_duplicates(subset=["WBS"])
    .set_index("WBS")["Name"]
    .to_dict()
)

phase_slip = (
    df.groupby("PhaseKey")["SlippageExposure"]
    .sum()
    .reset_index()
    .sort_values("SlippageExposure", ascending=False)
)

# Map PhaseKey → phase name where possible
phase_slip["PhaseName"] = phase_slip["PhaseKey"].map(phase_names).fillna(phase_slip["PhaseKey"])


# -----------------------------------------------------------
# EXEC SUMMARY TEXT
# -----------------------------------------------------------
top_phase = phase_slip.iloc[0] if not phase_slip.empty else None
second_phase = phase_slip.iloc[1] if len(phase_slip) > 1 else None

summary_parts = []

if slip_days > 1:
    summary_parts.append(f"Project is **{slip_days:.1f} days behind baseline**.")
elif slip_days < -1:
    summary_parts.append(f"Project is **{abs(slip_days):.1f} days ahead of baseline**.")
else:
    summary_parts.append("Project is roughly on its **baseline timeline**.")

if top_phase is not None:
    share = (top_phase["SlippageExposure"] / total_slip_exposure * 100) if total_slip_exposure > 0 else 0
    summary_parts.append(
        f"Most schedule risk is concentrated in **{top_phase['PhaseName']}** "
        f"(~{share:.0f}% of total slippage exposure)."
    )

if second_phase is not None:
    share2 = (second_phase["SlippageExposure"] / total_slip_exposure * 100) if total_slip_exposure > 0 else 0
    if share2 > 10:
        summary_parts.append(
            f"Secondary pressure comes from **{second_phase['PhaseName']}** "
            f"(~{share2:.0f}%)."
        )

if new_crit > 0:
    summary_parts.append(
        f"**{new_crit} tasks** were not critical in the baseline but are now critical in the live network."
    )

if behind_count > 0:
    summary_parts.append(
        f"**{behind_count} tasks** are currently behind their time-based expected progress."
    )

st.markdown("### Executive Summary")

st.markdown(
    " ".join(summary_parts)
    or "No obvious schedule horror shows yet. This is suspicious."
)

st.divider()

# -----------------------------------------------------------
# KPI ROW
# -----------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline Finish (Project Length)", f"{bl_finish:.1f} days")

with col2:
    delta_color = "normal"
    if slip_days > 1:
        delta_color = "inverse"  # red-ish in Streamlit
    elif slip_days < -1:
        delta_color = "normal"

    st.metric(
        "Live Finish (Current Plan)",
        f"{lv_finish:.1f} days",
        delta=f"{slip_days:+.1f} days vs baseline",
        delta_color=delta_color,
    )

with col3:
    st.metric(
        "Critical Tasks (Live)",
        crit_live,
        help="Tasks with effectively zero float in the live network."
    )


# -----------------------------------------------------------
# TOP VISUALS ROW
# -----------------------------------------------------------
left, right = st.columns(2)

# --- A. Slippage Exposure by Phase (Donut) ---
with left:
    st.markdown("#### Slippage Exposure by Phase")

    if phase_slip["SlippageExposure"].sum() <= 0:
        st.info("No slippage exposure detected yet.")
    else:
        fig_phase = px.pie(
            phase_slip,
            names="PhaseName",
            values="SlippageExposure",
            hole=0.45,
            color="PhaseName",
            color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig_phase.update_traces(textinfo="percent+label")
        fig_phase.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_phase, use_container_width=True)

# --- B. Criticality Distribution (Donut) ---
with right:
    st.markdown("#### Task Criticality (Live)")

    crit_count = int((df["IsCritical_LV"]).sum())
    near_count = int((df["IsNearCritical_LV"]).sum())
    safe_count = int(len(df) - crit_count - near_count)

    crit_data = pd.DataFrame(
        {
            "Status": ["Critical", "Near-Critical", "Safe"],
            "Count": [crit_count, near_count, safe_count],
        }
    )

    fig_crit = px.pie(
        crit_data,
        names="Status",
        values="Count",
        hole=0.5,
        color="Status",
        color_discrete_map={
            "Critical": FLAT_COLORS["red"],
            "Near-Critical": FLAT_COLORS["amber"],
            "Safe": FLAT_COLORS["green"],
        },
    )
    fig_crit.update_traces(textinfo="label+percent")
    fig_crit.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig_crit, use_container_width=True)

st.divider()

# -----------------------------------------------------------
# 🔥 SCHEDULE RISK BY OWNER
# -----------------------------------------------------------

st.subheader("🔥 Schedule Risk by Owner")

owner_slip = (
    df.groupby("Owner")["SlippageExposure"]
    .sum()
    .reset_index()
    .sort_values("SlippageExposure", ascending=False)
)

if owner_slip["SlippageExposure"].sum() <= 0:
    st.success("No meaningful slippage exposure attributed to any owner.")
else:
    # Horizontal bar chart
    fig_owner = px.bar(
        owner_slip,
        x="SlippageExposure",
        y="Owner",
        orientation="h",
        color="SlippageExposure",
        color_continuous_scale="RdYlGn_r",
        title="Which Owner Drives Most Schedule Risk?",
    )

    fig_owner.update_layout(
        xaxis_title="Slippage Exposure",
        yaxis_title="Owner",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_owner, use_container_width=True)

    # Donut: Ownership share
    st.markdown("#### Risk Share by Owner")

    fig_owner_pie = px.pie(
        owner_slip,
        names="Owner",
        values="SlippageExposure",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Bluered_r,
    )

    fig_owner_pie.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Exposure: %{value:.1f}<extra></extra>"
    )

    fig_owner_pie.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_owner_pie, use_container_width=True)

# -----------------------------------------------------------
# SECONDARY VISUALS
# -----------------------------------------------------------

col_a, col_b = st.columns(2)

# --- C. Top 10 Tasks Behind Schedule ---
with col_a:
    st.markdown("#### Top 10 Tasks Behind Schedule")

    if behind.empty:
        st.success("No tasks are behind schedule right now.")
    else:
        top10 = (
            behind.copy()
            .sort_values("ScheduleVariance")  # most negative first
            .head(10)
        )
        top10["TaskLabel"] = top10["TaskID"].astype(str) + " – " + top10["Name"].astype(str)

        fig_late = px.bar(
            top10,
            x="ScheduleVariance",
            y="TaskLabel",
            orientation="h",
            color_discrete_sequence=[FLAT_COLORS["red"]],
        )
        fig_late.update_layout(
            xaxis_title="Schedule Variance (Actual % - Expected %)",
            yaxis_title="Task",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_late, use_container_width=True)

# --- D. Risk Heatmap (Phase vs Outline Level) ---
with col_b:
    st.markdown("#### Risk Heatmap (Slippage Exposure)")

    leaf_df = df[df["IsLeaf"]].copy()
    if leaf_df["SlippageExposure"].sum() <= 0:
        st.info("No meaningful slippage exposure yet to build a heatmap.")
    else:
        heat_df = (
            leaf_df.groupby(["PhaseKey", "Outline Level"])["SlippageExposure"]
            .sum()
            .reset_index()
        )
        heat_df["Outline Level"] = heat_df["Outline Level"].astype(int)

        fig_heat = px.density_heatmap(
            heat_df,
            x="PhaseKey",
            y="Outline Level",
            z="SlippageExposure",
            color_continuous_scale="RdYlGn_r",
        )
        fig_heat.update_layout(
            xaxis_title="Phase",
            yaxis_title="Outline Level",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# # -----------------------------------------------------------
# # CRITICAL PATH VIEW (Live)
# # -----------------------------------------------------------
#
# st.markdown("### Live Critical Path View")
#
# cp_df = df[df["IsCritical_LV"]].copy()
#
# if cp_df.empty:
#     st.info("No live critical path found (no tasks with zero float).")
# else:
#     # Build a simple ordered view by live ES
#     cp_df = cp_df.sort_values("LV_ES")
#     cp_df["Order"] = range(1, len(cp_df) + 1)
#     cp_df["TaskLabel"] = cp_df["Order"].astype(str) + ". " + cp_df["Name"].astype(str)
#
#     fig_cp = px.line(
#         cp_df,
#         x="LV_ES",
#         y="LV_EF",
#         text="TaskLabel",
#         markers=True,
#         color_discrete_sequence=[FLAT_COLORS["blue"]],
#     )
#     fig_cp.update_traces(textposition="top center")
#     fig_cp.update_layout(
#         xaxis_title="Live ES (days)",
#         yaxis_title="Live EF (days)",
#         margin=dict(l=10, r=10, t=10, b=10),
#     )
#
#     st.plotly_chart(fig_cp, use_container_width=True)
#
# st.divider()
#
# # -----------------------------------------------------------
# # FULL DATA TABLE
# # -----------------------------------------------------------
# st.markdown("### Full CPM Output")
#
# with st.expander("Show full CPM table"):
#     display_cols = [
#         "TaskID", "Name", "WBS", "Outline Level",
#         "Dur_BL", "Duration",
#         "PercentComplete", "ExpectedPct",
#         "BL_ES", "BL_EF", "LV_ES", "LV_EF",
#         "Float_BL", "Float_LV",
#         "IsCritical_BL", "IsCritical_LV", "IsNearCritical_LV",
#         "ScheduleVariance", "Remaining_LV",
#         "SlippageExposure", "CriticalityWeight",
#     ]
#     existing_display_cols = [c for c in display_cols if c in df.columns]
#     st.dataframe(
#         df[existing_display_cols].sort_values(["WBS", "TaskID"]),
#         use_container_width=True,
#     )