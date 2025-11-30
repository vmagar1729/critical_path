# critical_path/pages/03_Analytics.py

import os
import sys

# -------------------------------------------------------------------
# Path bootstrap: same pattern as other pages
# -------------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df
from critical_path.cpm.analytics_engine import (
    ensure_analytics_fields,
    compute_kpis,
    add_float_bucket,
)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Analytics",
    layout="wide",
)

st.title("📊 Deep Project Analytics")

st.caption(
    "For when the executive summary isn't enough and you actually want to see how the machine behaves."
)

# -------------------------------------------------------------------
# Load schedule from session
# -------------------------------------------------------------------
df_raw = st.session_state.get("schedule_df", None)

if df_raw is None:
    st.warning("No schedule loaded. Upload one on the Home/Menu page first.")
    st.stop()

# Ensure datetime fields
for col in ["Start", "Finish", "Baseline Start", "Baseline Finish"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_datetime(df_raw[col], errors="coerce")

# Owner normalization
if "Owner" not in df_raw.columns:
    df_raw["Owner"] = "Unassigned"
df_raw["Owner"] = df_raw["Owner"].fillna("Unassigned").astype(str)

# -------------------------------------------------------------------
# Run CPM + Intelligence
# -------------------------------------------------------------------
try:
    df = compute_dual_cpm_from_df(df_raw)
except Exception as e:
    st.error(f"Error computing CPM: {e}")
    st.stop()

# Bring Owner across if the engine dropped it
if "Owner" not in df.columns and "TaskID" in df_raw.columns:
    owners = df_raw[["TaskID", "Owner"]].drop_duplicates()
    df = df.merge(owners, on="TaskID", how="left")

df["Owner"] = df["Owner"].fillna("Unassigned").astype(str)

# Enrich with analytics-specific fields
df = ensure_analytics_fields(df)
df = add_float_bucket(df)

# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("Filters")

owners = sorted(df["Owner"].dropna().unique().tolist())
selected_owners = st.sidebar.multiselect(
    "Owner",
    options=owners,
    default=owners,
)

if "Outline Level" in df.columns:
    min_lvl = int(df["Outline Level"].min())
    max_lvl = int(df["Outline Level"].max())
    sel_min, sel_max = st.sidebar.slider(
        "Outline Level range",
        min_value=min_lvl,
        max_value=max_lvl,
        value=(min_lvl, max_lvl),
        step=1,
    )
else:
    sel_min, sel_max = (None, None)

status_options = ["All", "On track / ahead", "Behind"]
sel_status = st.sidebar.selectbox("Status filter", status_options, index=0)

df_filt = df.copy()

df_filt = df_filt[df_filt["Owner"].isin(selected_owners)]

if sel_min is not None and sel_max is not None and "Outline Level" in df_filt.columns:
    df_filt = df_filt[
        (df_filt["Outline Level"] >= sel_min)
        & (df_filt["Outline Level"] <= sel_max)
    ]

if sel_status != "All":
    behind_mask = df_filt["ScheduleVariance"] < 0
    if sel_status == "Behind":
        df_filt = df_filt[behind_mask]
    else:
        df_filt = df_filt[~behind_mask]

# -------------------------------------------------------------------
# KPI strip
# -------------------------------------------------------------------
kpi = compute_kpis(df_filt)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Total tasks", kpi["total_tasks"])

with c2:
    st.metric("Tasks behind", kpi["behind_tasks"])

with c3:
    st.metric("Critical tasks", kpi["critical_tasks"])

with c4:
    st.metric("Avg % complete", f"{kpi['avg_percent_complete']:.1f}%")

with c5:
    st.metric("Remaining work (days)", f"{kpi['total_remaining']:.1f}")

st.divider()

# -------------------------------------------------------------------
# Layout: three analytical panels
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Progress & Health", "Float & Criticality", "Owner & WBS View"]
)

# -------------------------------------------------------------------
# TAB 1: Progress & Health
# -------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Planned vs Actual Progress")

        if "PercentComplete" in df_filt.columns and "ExpectedPct" in df_filt.columns:
            scatter_df = df_filt.copy()
            scatter_df["ExpectedPct"] = scatter_df["ExpectedPct"].clip(0, 100)
            scatter_df["PercentComplete"] = scatter_df["PercentComplete"].clip(0, 100)

            fig_scatter = px.scatter(
                scatter_df,
                x="ExpectedPct",
                y="PercentComplete",
                color="Owner",
                hover_data=["TaskID", "Name"],
                labels={
                    "ExpectedPct": "Planned % complete",
                    "PercentComplete": "Actual % complete",
                },
            )
            fig_scatter.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=100,
                y1=100,
                line=dict(dash="dash"),
            )
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("ExpectedPct / PercentComplete not available for this dataset.")

    with col_b:
        st.subheader("Status distribution")

        if "ScheduleVariance" in df_filt.columns:
            status_df = df_filt.copy()
            status_df["Status"] = np.where(
                status_df["ScheduleVariance"] < 0, "Behind", "On / Ahead"
            )
            status_counts = (
                status_df.groupby("Status")["TaskID"].count().reset_index()
            )

            if not status_counts.empty:
                fig_pie = px.pie(
                    status_counts,
                    names="Status",
                    values="TaskID",
                    hole=0.45,
                )
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No tasks in current filter selection.")
        else:
            st.info("ScheduleVariance not available for this dataset.")

    st.markdown("### Duration Creep")

    creep_df = df_filt[df_filt.get("HasDurationCreep", False)]

    if creep_df.empty:
        st.success("No tasks with increased duration vs baseline in this filter.")
    else:
        col_c1, col_c2 = st.columns([2, 3])

        with col_c1:
            st.metric(
                "Tasks with duration increase",
                len(creep_df),
            )
            st.metric(
                "Total added duration (days)",
                f"{creep_df['DurationCreep'].sum():.1f}",
            )

        with col_c2:
            top_creep = creep_df.sort_values("DurationCreep", ascending=False).head(15)
            fig_creep = px.bar(
                top_creep,
                x="DurationCreep",
                y="Name",
                color="Owner",
                orientation="h",
                labels={"DurationCreep": "Added days vs baseline"},
            )
            fig_creep.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_creep, use_container_width=True)

# -------------------------------------------------------------------
# TAB 2: Float & Criticality
# -------------------------------------------------------------------
with tab2:
    col_f1, col_f2 = st.columns([2, 3])

    with col_f1:
        st.subheader("Float distribution")

        if "FloatBucket" in df_filt.columns:
            fb = (
                df_filt.groupby("FloatBucket")["TaskID"]
                .count()
                .reset_index()
                .sort_values("TaskID", ascending=False)
            )
            fig_float = px.bar(
                fb,
                x="FloatBucket",
                y="TaskID",
                labels={"TaskID": "Tasks"},
                title="Tasks by live float bucket",
            )
            fig_float.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_float, use_container_width=True)
        else:
            st.info("Float information not available.")

    with col_f2:
        st.subheader("Slippage exposure vs float")

        if "SlippageExposure" in df_filt.columns and "Float_LV" in df_filt.columns:
            heat_df = df_filt.copy()
            heat_df["FloatBucket"] = heat_df.get("FloatBucket", "Unknown")

            agg = (
                heat_df.groupby(["Owner", "FloatBucket"])["SlippageExposure"]
                .sum()
                .reset_index()
            )

            if not agg.empty:
                fig_heat = px.density_heatmap(
                    agg,
                    x="FloatBucket",
                    y="Owner",
                    z="SlippageExposure",
                    color_continuous_scale="Reds",
                    labels={"SlippageExposure": "Exposure"},
                )
                fig_heat.update_layout(margin=dict(l=20, r=20, t=40, b=40))
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("No slippage exposure in current filter selection.")
        else:
            st.info("SlippageExposure / Float_LV missing.")

# -------------------------------------------------------------------
# TAB 3: Owner & WBS View
# -------------------------------------------------------------------
with tab3:
    st.subheader("Remaining work by Owner")

    if "Remaining_LV" in df_filt.columns:
        by_owner = (
            df_filt.groupby("Owner")["Remaining_LV"]
            .sum()
            .reset_index()
            .sort_values("Remaining_LV", ascending=False)
        )

        if not by_owner.empty:
            fig_owner = px.bar(
                by_owner,
                x="Owner",
                y="Remaining_LV",
                title="Remaining live duration by Owner",
                labels={"Remaining_LV": "Days"},
            )
            fig_owner.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_owner, use_container_width=True)
        else:
            st.info("No remaining work in current filter selection.")
    else:
        st.info("Remaining_LV missing; analytics engine did not compute it.")

    st.markdown("### WBS Treemap")

    if "WBS" in df_filt.columns and "SlippageExposure" in df_filt.columns:
        treemap_df = df_filt.copy()
        treemap_df["WBS"] = treemap_df["WBS"].astype(str)

        fig_tree = px.treemap(
            treemap_df,
            path=["WBS", "Owner", "Name"],
            values="SlippageExposure",
            color="Owner",
            title="Slippage exposure by WBS / Owner / Task",
        )
        fig_tree.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("WBS or SlippageExposure missing; treemap not available.")

# -------------------------------------------------------------------
# Nerd view
# -------------------------------------------------------------------
with st.expander("Nerd view: full analytics dataframe"):
    st.dataframe(df_filt, use_container_width=True)