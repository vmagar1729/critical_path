import os, sys

# Absolute directory containing Menu.py
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# The project root: Menu.py → critical_path → src
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# src/critical_path/pages/05_Predictive_Engine.py

from critical_path.bootstrap import *  # ensure imports & sys.path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df


st.set_page_config(page_title="Predictive Fix Engine", layout="wide")


# -------------------------------
# Helpers
# -------------------------------

def require_schedule_df():
    if "schedule_df" not in st.session_state or st.session_state["schedule_df"] is None:
        st.error("No schedule loaded. Go to **Menu** first and upload a CSV.")
        st.stop()
    return st.session_state["schedule_df"]


def compute_cpm_once(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full CPM + intelligence layer on the base normalized schedule.
    """
    df_cpm = compute_dual_cpm_from_df(base_df)
    return df_cpm


def simulate_compression(base_df: pd.DataFrame,
                         task_ids,
                         compression_pct: float) -> tuple[pd.DataFrame, float]:
    """
    Create a scenario where selected tasks have their Duration reduced
    by `compression_pct` %, recompute CPM, and return:
      - df_sim_cpm (CPM results on scenario)
      - new_lv_finish (float)
    """
    df_sim = base_df.copy()

    mask = df_sim["TaskID"].isin(task_ids)
    if not mask.any():
        return compute_dual_cpm_from_df(base_df), np.nan

    # Reduce Duration, but don't let it go below 0.5 days (or 0 if you want)
    df_sim.loc[mask, "Duration"] = (
        df_sim.loc[mask, "Duration"].astype(float) * (1 - compression_pct / 100.0)
    ).clip(lower=0.5)

    df_sim_cpm = compute_dual_cpm_from_df(df_sim)
    new_lv_finish = float(df_sim_cpm["LV_EF"].max())

    return df_sim_cpm, new_lv_finish


# -------------------------------
# Main UI
# -------------------------------

st.title("🧠 Predictive Fix Engine")
st.caption("“If this is late, whose work do we squeeze to get our lives back?”")

base_df = require_schedule_df()

with st.spinner("Running CPM engine on current schedule..."):
    df = compute_cpm_once(base_df)

# -------------------------------
# High-level slip picture
# -------------------------------

bl_finish = float(df["BL_EF"].max())
lv_finish = float(df["LV_EF"].max())
slip_days = lv_finish - bl_finish

top_row = st.columns(3)
with top_row[0]:
    st.metric("Baseline Finish (CPM)", f"{bl_finish:.1f} days")

with top_row[1]:
    st.metric("Live Finish (CPM)", f"{lv_finish:.1f} days",
              delta=f"{slip_days:+.1f} days")

with top_row[2]:
    crit_count = int(df["IsCritical_LV"].sum())
    st.metric("Live Critical Tasks", crit_count)

st.markdown("---")

# -------------------------------
# If there is no slip, say so and stop
# -------------------------------

if slip_days <= 0:
    st.success(
        "Your live schedule is **on or ahead of baseline**.\n\n"
        "You *can* still use this page to play “what-if compression,” "
        "but strictly speaking there’s nothing to rescue."
    )

# -------------------------------
# Candidate tasks to fix the schedule
# -------------------------------

st.subheader("1️⃣ Where is the pain concentrated?")

# Define "behind" tasks and critical / near-critical
behind = df[df["ScheduleVariance"] < 0].copy()

if behind.empty:
    st.info("No tasks are currently behind their expected baseline progress.")
else:
    # Prefer live critical & near-critical, sorted by SlippageExposure
    candidates = behind.copy()
    candidates = candidates[
        (candidates["IsCritical_LV"]) | (candidates["IsNearCritical_LV"])
    ].copy()

    if candidates.empty:
        st.warning(
            "Tasks behind schedule exist, but none are on or near the live critical path.\n"
            "Compressing them will mostly help local teams, not the project finish date."
        )
        candidates = behind.copy()

    # Owner-safe: some schedules may not have this column
    owner_col = "Owner" if "Owner" in candidates.columns else None
    sort_cols = ["SlippageExposure"]
    candidates = candidates.sort_values(sort_cols, ascending=False)

    st.markdown("**Top risk tasks (ranked by Slippage Exposure):**")

    show_cols = ["TaskID", "Name", "Remaining_LV",
                 "SlippageExposure", "ScheduleVariance",
                 "IsCritical_LV", "IsNearCritical_LV"]
    if owner_col:
        show_cols.insert(2, owner_col)

    st.dataframe(
        candidates[show_cols].head(20),
        use_container_width=True
    )

    # Pretty treemap by Owner if available
    if owner_col:
        st.markdown("#### Slippage Exposure by Owner (Live)")
        agg = candidates.groupby(owner_col, as_index=False)["SlippageExposure"].sum()
        agg = agg[agg["SlippageExposure"] > 0]

        if not agg.empty:
            fig = px.treemap(
                agg,
                path=[owner_col],
                values="SlippageExposure",
                title="Who carries the most schedule risk right now?",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positive Slippage Exposure yet. Everyone is either on track or not started.")

st.markdown("---")

# -------------------------------
# Recovery scenario controls
# -------------------------------

st.subheader("2️⃣ Build a schedule recovery scenario")

left, right = st.columns([2, 1])

with right:
    st.markdown("**Recovery knobs**")

    max_tasks = st.slider(
        "Max number of tasks to compress",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="We’ll pick the top N high-impact tasks."
    )

    compression_pct = st.slider(
        "Compression per task (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Rough reduction of remaining duration for selected tasks."
    )

    target_recovery = st.slider(
        "Desired recovery (days)",
        min_value=0.0,
        max_value=max(0.0, float(abs(slip_days)) + 10.0),
        value=max(0.0, float(abs(slip_days))),
        step=1.0,
        help="How many days of slip you want to claw back."
    )

    run_button = st.button("🔮 Suggest Recovery Plan", type="primary")

with left:
    st.markdown(
        """
        This engine will:
        1. Take your current live CPM results  
        2. Rank tasks by **SlippageExposure** × criticality  
        3. Compress the top N tasks by the chosen %  
        4. Recompute CPM and show the new finish date  
        5. Tell you whether that gets you close to your target
        """
    )

st.markdown("---")

# -------------------------------
# Run simulation
# -------------------------------

if run_button:
    if "SlippageExposure" not in df.columns or "Remaining_LV" not in df.columns:
        st.error("Schedule is missing intelligence metrics. Run through CPM engine first.")
        st.stop()

    # Use same candidate logic as above
    candidates = df.copy()
    candidates = candidates[df["Remaining_LV"] > 0].copy()
    candidates = candidates[
        (candidates["IsCritical_LV"]) | (candidates["IsNearCritical_LV"])
    ].copy()

    if candidates.empty:
        st.warning(
            "No critical / near-critical tasks with remaining work.\n"
            "Either this schedule is already done, or the data is lying to you."
        )
        st.stop()

    candidates = candidates.sort_values(
        ["SlippageExposure", "Remaining_LV"],
        ascending=False
    )

    selected = candidates.head(max_tasks).copy()
    selected_ids = list(selected["TaskID"])

    df_sim_cpm, new_lv_finish = simulate_compression(base_df, selected_ids, compression_pct)

    if np.isnan(new_lv_finish):
        st.error("Simulation failed for some reason. Check input data.")
        st.stop()

    recovered = lv_finish - new_lv_finish
    residual_slip = (new_lv_finish - bl_finish)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("New Live Finish", f"{new_lv_finish:.1f} days")
    with k2:
        st.metric("Recovery vs Current", f"{recovered:.1f} days")
    with k3:
        st.metric("Slip vs Baseline (after fix)", f"{residual_slip:+.1f} days")

    st.markdown("### Recommended Compression Set")

    show_cols = ["TaskID", "Name", "Remaining_LV",
                 "SlippageExposure", "IsCritical_LV", "IsNearCritical_LV"]
    if "Owner" in selected.columns:
        show_cols.insert(2, "Owner")

    selected["CompressionPct"] = compression_pct
    st.dataframe(
        selected[show_cols + ["CompressionPct"]],
        use_container_width=True
    )

    # Bar chart: remaining vs compressed
    chart_df = selected.copy()
    chart_df["Remaining_After"] = chart_df["Remaining_LV"] * (1 - compression_pct / 100.0)

    fig2 = px.bar(
        chart_df,
        x="Name",
        y=["Remaining_LV", "Remaining_After"],
        barmode="group",
        title="Remaining Effort Before vs After Compression (Selected Tasks)"
    )
    fig2.update_layout(xaxis_title="Task", yaxis_title="Effort (days)")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        **Interpretation for Execs:**

        - These tasks are the *least bad* place to apply pressure.  
        - If teams can realistically deliver the selected compression,  
          this is the new expected finish.  
        - If not, you’re not negotiating with physics, you’re negotiating with fantasy.
        """
    )
else:
    st.info("Set your knobs on the right and click **“Suggest Recovery Plan”** to see a what-if scenario.")