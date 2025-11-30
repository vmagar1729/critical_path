# critical_path/pages/04_Prediction_Engine.py

import os
import sys

# -------------------------------------------------------------------
# Path bootstrap (same pattern as other pages)
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
from critical_path.cpm.prediction_engine import (
    ensure_prediction_ready,
    run_monte_carlo,
    compute_slack_burndown,
    compute_owner_overload,
    cluster_risks,
    generate_executive_narrative,
)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Prediction Engine",
    layout="wide",
)

st.title("🔮 Prediction & Risk Engine")

st.caption(
    "Forward-looking view of the plan: Monte-Carlo dates, risk clusters, slack burn-down, "
    "owner overload, and a generated management narrative."
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

if "Owner" not in df_raw.columns:
    df_raw["Owner"] = "Unassigned"
df_raw["Owner"] = df_raw["Owner"].fillna("Unassigned").astype(str)

# -------------------------------------------------------------------
# Run CPM + intelligence
# -------------------------------------------------------------------
try:
    df_cpm = compute_dual_cpm_from_df(df_raw)
except Exception as e:
    st.error(f"Error computing CPM for prediction engine: {e}")
    st.stop()

# Bring Owner back if needed
if "Owner" not in df_cpm.columns and "TaskID" in df_raw.columns:
    owners = df_raw[["TaskID", "Owner"]].drop_duplicates()
    df_cpm = df_cpm.merge(owners, on="TaskID", how="left")

df_cpm = ensure_prediction_ready(df_cpm)

# -------------------------------------------------------------------
# Top-level tabs
# -------------------------------------------------------------------
tabs = st.tabs(
    [
        "Monte-Carlo Simulation",
        "Risk Forecasting",
        "Slack Burndown Projection",
        "Owner Overload Forecast",
        "Critical Path View",
        "AI-Generated Narrative",
    ]
)

# ===================================================================
# 1. Monte-Carlo Simulation
# ===================================================================
with tabs[0]:
    st.subheader("🎲 Monte-Carlo Finish Date Simulation")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        iters = st.slider("Number of simulation runs", 200, 3000, 800, step=200)
        crit_thresh = st.slider(
            "Include tasks with live float ≤",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.5,
            help="Only tasks with Float_LV at or below this threshold are treated as controlling the date.",
        )

    try:
        mc = run_monte_carlo(df_cpm, n_iter=iters, critical_float_threshold=crit_thresh)
    except Exception as e:
        st.error(f"Monte-Carlo simulation failed: {e}")
        st.stop()

    bl = mc["baseline_finish"]
    lv = mc["live_finish"]
    samples = mc["samples"]

    slip_mean = float(np.mean(mc["slip_samples"]))
    p50 = mc["p50"]
    p80 = mc["p80"]
    p90 = mc["p90"]
    prob_on_or_before = mc["prob_on_or_before_baseline"]

    with col_left:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline finish (day index)", f"{bl:.1f}")
        with c2:
            st.metric("Live finish (deterministic)", f"{lv:.1f}")
        with c3:
            st.metric("Mean simulated slip vs baseline", f"{slip_mean:+.1f} days")

        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("P50 finish (day)", f"{p50:.1f}")
        with c5:
            st.metric("P80 finish (day)", f"{p80:.1f}")
        with c6:
            st.metric("P90 finish (day)", f"{p90:.1f}")

        st.markdown(
            f"**Probability of finishing on or before baseline:** "
            f"~**{prob_on_or_before * 100:.0f}%**"
        )

    # Histogram of simulated finishes
    hist_df = pd.DataFrame({"FinishDay": samples})
    fig_hist = px.histogram(
        hist_df,
        x="FinishDay",
        nbins=40,
        title="Distribution of simulated finish days",
    )
    fig_hist.add_vline(x=bl, line_dash="dash", line_color="green", annotation_text="Baseline")
    fig_hist.add_vline(x=lv, line_dash="dot", line_color="orange", annotation_text="Live")
    fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Gatekeeper tasks used in simulation"):
        gate = mc["used_tasks"].copy()
        if gate.empty:
            st.info("No tasks qualified as controlling the date at the chosen float threshold.")
        else:
            st.dataframe(
                gate[
                    [
                        "TaskID",
                        "Name",
                        "Owner",
                        "Remaining_LV",
                        "Float_LV",
                        "SlippageExposure",
                    ]
                ].sort_values("SlippageExposure", ascending=False),
                use_container_width=True,
            )

# ===================================================================
# 2. Risk Forecasting (clusters)
# ===================================================================
with tabs[1]:
    st.subheader("⚠️ Risk Forecasting & Clusters")

    try:
        df_risk, cluster_info = cluster_risks(df_cpm, n_clusters=3)
    except Exception as e:
        st.error(f"Risk clustering failed: {e}")
        st.stop()

    # Map cluster index to label
    label_map = {
        0: "High risk",
        1: "Medium risk",
        2: "Low risk",
    }

    df_risk["RiskBand"] = df_risk["RiskCluster"].map(label_map).fillna("Unclassified")

    # Summary counts
    band_counts = (
        df_risk.groupby("RiskBand")["TaskID"]
        .count()
        .reset_index()
        .rename(columns={"TaskID": "Tasks"})
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("### Risk bucket counts")
        st.dataframe(band_counts, use_container_width=True)

    with c2:
        st.markdown("### Remaining exposure by risk band")
        if "SlippageExposure" in df_risk.columns:
            band_exp = (
                df_risk.groupby("RiskBand")["SlippageExposure"]
                .sum()
                .reset_index()
            )
            fig_band = px.bar(
                band_exp,
                x="RiskBand",
                y="SlippageExposure",
                title="Total modeled slippage exposure by risk band",
            )
            fig_band.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_band, use_container_width=True)
        else:
            st.info("SlippageExposure not available; cluster exposure view disabled.")

    st.markdown("### Task-level risk map")

    if "SlippageExposure" in df_risk.columns:
        scatter_df = df_risk.copy()
        scatter_df["Remaining_LV"] = pd.to_numeric(
            scatter_df["Remaining_LV"], errors="coerce"
        ).fillna(0.0)
        scatter_df["SlippageExposure"] = pd.to_numeric(
            scatter_df["SlippageExposure"], errors="coerce"
        ).fillna(0.0)

        fig_sc = px.scatter(
            scatter_df,
            x="Remaining_LV",
            y="SlippageExposure",
            color="RiskBand",
            hover_data=["TaskID", "Name", "Owner"],
            labels={
                "Remaining_LV": "Remaining duration (days)",
                "SlippageExposure": "Modeled slippage exposure",
            },
            title="Risk clusters by remaining work and slippage exposure",
        )
        fig_sc.update_layout(margin=dict(l=20, r=20, t=40, b=40))
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Missing SlippageExposure; cannot build risk scatter.")

    st.markdown("### Highest-risk tasks (cluster 0)")

    top_risk = (
        df_risk[df_risk["RiskCluster"] == 0]
        .copy()
        .sort_values("SlippageExposure", ascending=False)
        .head(20)
    )
    if top_risk.empty:
        st.info("No tasks fell into the highest-risk cluster.")
    else:
        st.dataframe(
            top_risk[
                [
                    "TaskID",
                    "Name",
                    "Owner",
                    "Remaining_LV",
                    "Float_LV",
                    "SlippageExposure",
                    "ScheduleVariance",
                ]
            ],
            use_container_width=True,
        )

# ===================================================================
# 3. Slack Burndown Projection
# ===================================================================
with tabs[2]:
    st.subheader("⏳ Slack Burn-down Projection")

    burn_df = compute_slack_burndown(df_cpm, bins=12)

    if burn_df.empty:
        st.info("Unable to compute slack burn-down for this dataset.")
    else:
        fig_burn = px.line(
            burn_df,
            x="ProgressPct",
            y="RemainingFloat",
            markers=True,
            labels={
                "ProgressPct": "Approx. project progress (%)",
                "RemainingFloat": "Remaining float (sum of positive live float)",
            },
            title="Projected slack consumption vs project progress",
        )
        fig_burn.update_layout(margin=dict(l=20, r=20, t=40, b=40))
        st.plotly_chart(fig_burn, use_container_width=True)

        st.markdown(
            """
            This curve tells you how quickly your safety margin disappears as the plan advances.
            Left side = early in the project; right side = late.
            A steep drop near the middle or end is a classic “all slack burned late” anti-pattern.
            """
        )

# ===================================================================
# 4. Owner Overload Forecast
# ===================================================================
with tabs[3]:
    st.subheader("👥 Owner Overload Forecast")

    overload_df = compute_owner_overload(df_cpm, critical_float_threshold=0.5)

    if overload_df.empty:
        st.info("No remaining work found; overload forecast is trivial (everyone is done).")
    else:
        st.markdown("### Overload index by owner")

        fig_ov = px.bar(
            overload_df,
            x="Owner",
            y="OverloadIndex",
            hover_data=["TotalRemaining", "CriticalRemaining"],
            title="Owner overload index (critical / total remaining work)",
        )
        fig_ov.update_layout(margin=dict(l=20, r=20, t=40, b=40))
        st.plotly_chart(fig_ov, use_container_width=True)

        st.dataframe(overload_df, use_container_width=True)

        st.markdown(
            "High overload index = most of that owner’s remaining work sits on critical or near-critical paths."
        )

# ===================================================================
# 5. Critical Path View / Proto-animation
# ===================================================================
with tabs[4]:
    st.subheader("📐 Critical Path View")

    # We approximate an "animation" using a timeline colored by criticality.
    if "BL_ES" in df_cpm.columns and "Dur_BL" in df_cpm.columns:
        df_tl = df_cpm.copy()
        df_tl["StartDay"] = df_tl["BL_ES"]
        df_tl["FinishDay"] = df_tl["BL_ES"] + df_tl["Dur_BL"]
        df_tl["CriticalFlag"] = np.where(df_tl.get("IsCritical_LV", False), "Critical", "Non-critical")

        # Restrict to leaf tasks if we have that flag
        if "IsLeaf" in df_tl.columns:
            df_tl = df_tl[df_tl["IsLeaf"]]

        fig_tl = px.timeline(
            df_tl,
            x_start="StartDay",
            x_end="FinishDay",
            y="Name",
            color="CriticalFlag",
            hover_data=["TaskID", "Owner", "Float_LV"],
            title="Baseline timeline with live criticality overlay",
        )
        fig_tl.update_yaxes(autorange="reversed")
        fig_tl.update_layout(margin=dict(l=20, r=20, t=40, b=40))
        st.plotly_chart(fig_tl, use_container_width=True)

        st.markdown(
            "This is the static view behind a future critical-path animation: red bars are on the "
            "current live critical chain; other tasks can slip without immediately moving the finish."
        )
    else:
        st.info("BL_ES / Dur_BL not available; cannot render critical path timeline.")

# ===================================================================
# 6. AI-Generated Narrative (offline template)
# ===================================================================
with tabs[5]:
    st.subheader("📝 AI-Style Executive Narrative")

    # Reuse Monte-Carlo result and overload / risk views
    try:
        mc_for_narrative = run_monte_carlo(df_cpm, n_iter=800, critical_float_threshold=0.5)
    except Exception:
        mc_for_narrative = {
            "baseline_finish": float(df_cpm.get("BL_EF", pd.Series([0])).max()),
            "live_finish": float(df_cpm.get("LV_EF", pd.Series([0])).max()),
            "slip_samples": np.array([0.0]),
            "p80": float(df_cpm.get("LV_EF", pd.Series([0])).max()),
            "p90": float(df_cpm.get("LV_EF", pd.Series([0])).max()),
            "prob_on_or_before_baseline": 0.0,
        }

    overload_for_narr = compute_owner_overload(df_cpm, critical_float_threshold=0.5)

    try:
        df_risk_narr, _ = cluster_risks(df_cpm, n_clusters=3)
    except Exception:
        df_risk_narr = df_cpm.copy()
        df_risk_narr["RiskCluster"] = 1  # neutral

    narrative = generate_executive_narrative(
        mc_for_narrative,
        overload_for_narr,
        df_risk_narr,
    )

    st.markdown(narrative)