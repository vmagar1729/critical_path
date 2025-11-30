import os, sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df
from critical_path.cpm.executive_engine import compute_executive_metrics


# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(page_title="Executive Overview", layout="wide")
st.title("🧭 Executive Project Overview")

st.caption("Minimal, visual, sponsor-friendly summary. No jargon. No clutter.")


# -----------------------------------------------------------
# Load schedule
# -----------------------------------------------------------
df_raw = st.session_state.get("schedule_df", None)

if df_raw is None:
    st.warning("No schedule loaded. Upload one on the Home page.")
    st.stop()

for c in ["Start", "Finish", "Baseline Start", "Baseline Finish"]:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

if "Owner" not in df_raw.columns:
    df_raw["Owner"] = "Unassigned"


# -----------------------------------------------------------
# Run CPM Engine
# -----------------------------------------------------------
try:
    df = compute_dual_cpm_from_df(df_raw)
except Exception as e:
    st.error(f"Error computing CPM: {e}")
    st.stop()


# -----------------------------------------------------------
# Compute executive metrics
# -----------------------------------------------------------
metrics = compute_executive_metrics(df)

bl_finish = metrics["bl_finish"]
lv_finish = metrics["lv_finish"]
slip = metrics["slip"]


# -----------------------------------------------------------
# SECTION 1: Are we on track?
# -----------------------------------------------------------
st.markdown("## 1. Overall Status")

c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    st.metric("Baseline Finish (days)", f"{bl_finish:.1f}")

with c2:
    st.metric("Live Finish (days)", f"{lv_finish:.1f}",
              delta=f"{slip:+.1f} days")

with c3:
    st.markdown(
        f"""
        ### {metrics['status_color']} {metrics['status_label']}
        {metrics['status_desc']}
        """
    )

st.divider()


# -----------------------------------------------------------
# SECTION 2: If behind — Where? & Who owns it?
# -----------------------------------------------------------
st.markdown("## 2. Delay Location & Accountability")

behind_df = metrics["behind_df"].copy()
owner_group = metrics["owner_exposure_df"].copy()

if behind_df.empty:
    st.success("No tasks behind schedule. Carry on.")
else:
    col_pie, col_table = st.columns([2, 3])

    with col_pie:
        st.markdown("### Slippage Exposure by Owner")
        fig = px.pie(
            owner_group,
            names="Owner",
            values="TotalExposure",
            color="Owner",
            hole=0.45,
            hover_data=["TasksBehind", "MaxSlip"],
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("### Owner Accountability Table")
        show = owner_group.copy()
        show["TotalExposure"] = show["TotalExposure"].round(1)
        show["ExposurePct"] = show["ExposurePct"].round(1)
        show["MaxSlip"] = show["MaxSlip"].round(1)
        st.dataframe(show, use_container_width=True)

    st.markdown("### Top Delayed Tasks")

    top = behind_df.copy()
    top["SlipDays"] = (
        (top["ScheduleVariance"] / 100.0) * top["Dur_BL"]
    ).fillna(0)
    top = top.sort_values("SlipDays").head(10)

    fig2 = px.bar(
        top,
        x="SlipDays",
        y="Name",
        color="Owner",
        orientation="h",
        title="Worst Offenders",
        labels={"SlipDays": "Estimated Days Behind"},
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()


# -----------------------------------------------------------
# SECTION 3: Duration Creep
# -----------------------------------------------------------
st.markdown("## 3. Duration Creep (The silent killer)")

creep_df = metrics["creep_df"]

cA, cB, cC = st.columns(3)

with cA:
    st.metric("Tasks with Increased Duration", len(creep_df))

with cB:
    st.metric("Total Added Duration", f"{creep_df['DurationCreep'].sum():.1f} days")

with cC:
    m = creep_df["DurationCreep"].max() if len(creep_df) else 0
    st.metric("Max Single-Task Creep", f"{m:.1f} days")

if len(creep_df) > 0:
    st.dataframe(
        creep_df.sort_values("DurationCreep", ascending=False)[
            ["TaskID", "Name", "Owner", "Dur_BL", "Duration", "DurationCreep"]
        ],
        use_container_width=True
    )
else:
    st.success("No duration creep detected.")

st.divider()


# -----------------------------------------------------------
# SECTION 4: Recovery Feasibility (High-level only)
# -----------------------------------------------------------
st.markdown("## 4. Can We Recover?")

total_recoverable = metrics["total_recoverable"]

cR1, cR2 = st.columns([1, 3])

with cR1:
    st.metric("Slip (days)", f"{slip:.1f}")
    st.metric("Recoverable Float", f"{total_recoverable:.1f}")

with cR2:
    if slip <= 0.5:
        st.success("No recovery needed.")
    elif total_recoverable >= slip:
        st.success("Delay is recoverable using existing float.")
    elif 0 < total_recoverable < slip:
        st.warning(
            f"We can recover ~{total_recoverable:.1f} days, but "
            f"{slip - total_recoverable:.1f} additional days require mitigation."
        )
    else:
        st.error("Not recoverable without re-sequencing or scope decisions.")