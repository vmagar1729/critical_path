import os, sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df
from critical_path.cpm.recovery_engine import compute_recovery_plan


# -----------------------------------------------------------
# Page Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="Fix-the-Schedule Engine",
    layout="wide",
)

st.title("🧠 Predictive Fix-the-Schedule Engine")


# -----------------------------------------------------------
# Load schedule
# -----------------------------------------------------------
df_raw = st.session_state.get("schedule_df", None)

if df_raw is None:
    st.warning("No schedule loaded. Upload one on the Home page.")
    st.stop()

for col in ["Start", "Finish", "Baseline Start", "Baseline Finish"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_datetime(df_raw[col], errors="coerce")

if "Owner" not in df_raw.columns:
    df_raw["Owner"] = "Unassigned"


# -----------------------------------------------------------
# Run CPM
# -----------------------------------------------------------
try:
    df = compute_dual_cpm_from_df(df_raw)
except Exception as e:
    st.error(f"Error computing CPM: {e}")
    st.stop()


# Remerge Owner if needed
if "Owner" not in df.columns:
    df = df.merge(df_raw[["TaskID", "Owner"]], on="TaskID", how="left")


# -----------------------------------------------------------
# Compute recovery plan
# -----------------------------------------------------------
plan = compute_recovery_plan(df)
slip = plan["slip"]

bl_finish = df["BL_EF"].max()
lv_finish = df["LV_EF"].max()


# -----------------------------------------------------------
# Summary row
# -----------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Baseline Finish (days)", f"{bl_finish:.1f}")

with c2:
    st.metric("Live Finish (days)", f"{lv_finish:.1f}")

with c3:
    st.metric("Slip", f"{slip:+.1f} days")


st.divider()


# -----------------------------------------------------------
# If no recovery needed
# -----------------------------------------------------------
if slip <= 0:
    st.success("Project is not behind baseline. No recovery needed.")
    st.stop()


# -----------------------------------------------------------
# Show recovery scenarios
# -----------------------------------------------------------
scenario = plan["scenario_type"]
gate = plan["gatekeepers"]

if scenario == "impossible":
    st.error("Impossible to recover using current durations or float. Requires reprioritization or re-planning.")
    st.stop()


st.subheader("Recovery Feasibility")


if scenario == "single":

    t = plan["selected_tasks"].iloc[0]

    st.markdown(
        f"""
        ### 🎯 Single Task Can Recover Entire Slip

        - **Task:** `{int(t['TaskID'])} – {t['Name']}`
        - **Owner:** `{t['Owner']}`
        - **Remaining Duration:** `{t['Remaining_LV']:.1f}` days  
        - **Required Cut:** `{t['RequiredCut']:.1f}` days  
        - **New Remaining:** `{t['NewRemaining']:.1f}` days  
        """
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Current Remaining", "After Acceleration"],
        y=[t["Remaining_LV"], t["NewRemaining"]],
        marker_color=["indianred", "seagreen"]
    ))
    fig.update_layout(
        title="Single-Task Acceleration Impact",
        yaxis_title="Days"
    )

    st.plotly_chart(fig, use_container_width=True)

elif scenario == "multi":

    st.markdown("### 🔧 Multi-Task Recovery (Minimal Required Set)")

    sel = plan["selected_tasks"].copy()
    sel = sel.sort_values("RequiredCut", ascending=False)

    st.dataframe(
        sel[
            ["TaskID", "Name", "Owner", "Remaining_LV", "RequiredCut", "NewRemaining"]
        ],
        use_container_width=True
    )

    # Owner bar
    by_owner = sel.groupby("Owner")["RequiredCut"].sum().reset_index()

    fig = px.bar(
        by_owner,
        x="Owner",
        y="RequiredCut",
        title="Required Acceleration by Owner (Days)",
        color="Owner"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Waterfall
    fig2 = go.Figure(go.Waterfall(
        name="",
        measure=["absolute", "relative", "relative"],
        x=["Baseline", "Slip", "Recovery"],
        y=[bl_finish, slip, -slip],
    ))
    fig2.update_layout(title="Finish Date Recovery Simulation")

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No structured scenario available.")


st.divider()


# -----------------------------------------------------------
# Gatekeepers overview
# -----------------------------------------------------------
st.subheader("Gatekeepers (Tasks That Control the Date)")

if gate is not None and not gate.empty:
    st.dataframe(
        gate[
            ["TaskID", "Name", "Owner", "Remaining_LV",
             "Float_LV", "CriticalityWeight", "ImpactFactor"]
        ].sort_values("ImpactFactor", ascending=False),
        use_container_width=True
    )

    fig = px.treemap(
        gate,
        path=["Owner", "Name"],
        values="ImpactFactor",
        color="CriticalityWeight",
        color_continuous_scale="Reds",
        title="Gatekeeper Impact Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No gatekeepers found.")