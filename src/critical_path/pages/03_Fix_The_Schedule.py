import os
import sys
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------
# Path setup so `critical_path` is importable
# -----------------------------------------------------------

THIS_FILE = os.path.abspath(__file__)
PROJECT_SRC = os.path.abspath(os.path.join(THIS_FILE, "../../../"))

if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df


# -----------------------------------------------------------
# Helper: Compute recovery plan
# -----------------------------------------------------------

def compute_recovery_plan(df: pd.DataFrame,
                          float_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Given a CPM-enriched dataframe (from compute_dual_cpm_from_df),
    compute slip and a recommended recovery plan.

    Returns dict with keys:
      - slip
      - recoverable (bool)
      - gatekeepers (df)
      - scenario_type ("none" | "single" | "multi" | "impossible")
      - single_task (Series or None)
      - selected_tasks (DataFrame or None)
    """

    # Project finishes in "days from 0", not dates
    bl_finish = df["BL_EF"].max()
    lv_finish = df["LV_EF"].max()

    if pd.isna(bl_finish) or pd.isna(lv_finish):
        return {
            "slip": 0.0,
            "recoverable": False,
            "scenario_type": "none",
            "gatekeepers": df.head(0),
            "single_task": None,
            "selected_tasks": None,
        }

    slip = float(lv_finish - bl_finish)

    # If not behind, nothing to recover
    if slip <= 0:
        return {
            "slip": slip,
            "recoverable": False,
            "scenario_type": "none",
            "gatekeepers": df.head(0),
            "single_task": None,
            "selected_tasks": None,
        }

    # Gatekeepers: tasks that can actually move the finish
    # Use live float and only leaf tasks (if IsLeaf present)
    gate = df.copy()
    if "IsLeaf" in gate.columns:
        gate = gate[gate["IsLeaf"]]

    gate = gate[gate["Float_LV"] <= float_threshold]

    # Require positive remaining duration
    gate = gate[gate["Remaining_LV"] > 0]

    if gate.empty:
        # No tasks with leverage
        return {
            "slip": slip,
            "recoverable": False,
            "scenario_type": "impossible",
            "gatekeepers": df.head(0),
            "single_task": None,
            "selected_tasks": None,
        }

    # Compute impact factor: Remaining * CriticalityWeight
    gate = gate.copy()
    if "CriticalityWeight" not in gate.columns:
        # Defensive fallback
        gate["CriticalityWeight"] = np.where(gate["Float_LV"] == 0, 1.0, 0.5)

    gate["ImpactFactor"] = gate["Remaining_LV"] * gate["CriticalityWeight"]

    # Sort by impact desc
    gate = gate.sort_values("ImpactFactor", ascending=False)

    # Total possible recovery if you killed all remaining on gatekeepers
    total_possible = gate["Remaining_LV"].sum()

    if total_possible < slip - 1e-6:
        # Even if you annihilate everything, you can't fully recover
        return {
            "slip": slip,
            "recoverable": False,
            "scenario_type": "impossible",
            "gatekeepers": gate,
            "single_task": None,
            "selected_tasks": None,
        }

    # Scenario 1: Can a single task recover the entire slip?
    top = gate.iloc[0]
    if top["Remaining_LV"] >= slip:
        # Single-task recovery
        single = top.copy()
        single["RequiredCut"] = slip
        single["NewRemaining"] = top["Remaining_LV"] - slip
        single_df = pd.DataFrame([single])

        return {
            "slip": slip,
            "recoverable": True,
            "scenario_type": "single",
            "gatekeepers": gate,
            "single_task": single_df.iloc[0],
            "selected_tasks": single_df,
        }

    # Scenario 2: Multi-task minimal set (greedy)
    need = slip
    rows = []
    for _, r in gate.iterrows():
        if need <= 0:
            break
        max_cut = r["Remaining_LV"]
        cut = min(max_cut, need)
        need -= cut

        rr = r.copy()
        rr["RequiredCut"] = cut
        rr["NewRemaining"] = r["Remaining_LV"] - cut
        rows.append(rr)

    selected = pd.DataFrame(rows)

    return {
        "slip": slip,
        "recoverable": True,
        "scenario_type": "multi",
        "gatekeepers": gate,
        "single_task": None,
        "selected_tasks": selected,
    }


# -----------------------------------------------------------
# Streamlit Layout
# -----------------------------------------------------------

st.set_page_config(
    page_title="Fix-the-Schedule Engine",
    layout="wide",
)

st.title("🧠 Predictive Fix-the-Schedule Engine")

st.caption(
    "Answers three questions: "
    "1) Are we behind? 2) Who is responsible? 3) What do we change to recover the date?"
)

# uploaded = st.file_uploader("Upload Schedule CSV (same format as the other pages)", type=["csv"])
#
# if uploaded is None:
#     st.info("Upload a CSV exported from MS Project or your synthetic generator to see recovery options.")
#     st.stop()
#
# # Read CSV
# try:
#     df_raw = pd.read_csv(uploaded)
# except Exception as e:
#     st.error(f"Error reading CSV: {e}")
#     st.stop()

# Use the central session schedule
df_raw = st.session_state.get("schedule_df", None)

# Make sure key date columns are datetime
for col in ["Start", "Finish", "Baseline Start", "Baseline Finish"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_datetime(df_raw[col], errors="coerce")

# Owner column optional
if "Owner" not in df_raw.columns:
    df_raw["Owner"] = "Unassigned"

# Run CPM engine
try:
    df = compute_dual_cpm_from_df(df_raw)
except Exception as e:
    st.error(f"Error computing CPM: {e}")
    st.stop()

# Align "Owner" with df after engine (if engine dropped/changed columns, merge it back)
if "Owner" not in df.columns and "TaskID" in df_raw.columns:
    owners = df_raw[["TaskID", "Owner"]].drop_duplicates()
    df = df.merge(owners, on="TaskID", how="left")

if "Owner" not in df.columns:
    df["Owner"] = "Unassigned"

# -----------------------------------------------------------
# Recovery computation
# -----------------------------------------------------------

plan = compute_recovery_plan(df)
slip = plan["slip"]

bl_finish = df["BL_EF"].max()
lv_finish = df["LV_EF"].max()

# -----------------------------------------------------------
# Top row summary
# -----------------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Baseline Finish (days)", f"{bl_finish:.1f}" if pd.notnull(bl_finish) else "N/A")

with c2:
    st.metric("Live Finish (days)", f"{lv_finish:.1f}" if pd.notnull(lv_finish) else "N/A")

with c3:
    status = "On Track" if slip <= 0 else "Behind"
    delta_str = f"{slip:+.1f} days"
    st.metric("Schedule Status", status, delta=delta_str)

st.divider()

# If we're not behind, show a calm message and some context
if slip <= 0:
    st.success("Project is not behind baseline finish. No recovery actions required.")
    # Optional: show owners with remaining work anyway
    if "Remaining_LV" in df.columns:
        by_owner = (
            df.groupby("Owner", dropna=False)["Remaining_LV"]
            .sum()
            .reset_index()
            .sort_values("Remaining_LV", ascending=False)
        )
        st.subheader("Remaining Work by Owner")
        fig = px.bar(
            by_owner,
            x="Owner",
            y="Remaining_LV",
            title="Remaining Live Duration by Owner",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.stop()

# -----------------------------------------------------------
# We are behind → show recovery intelligence
# -----------------------------------------------------------

if not plan["recoverable"] and plan["scenario_type"] == "impossible":
    st.error(
        "This schedule cannot be fully recovered by accelerating the remaining work "
        "on the current critical/near-critical path. "
        "You need **scope changes or a new target date**."
    )
else:
    st.subheader("Recovery Feasibility")

    colA, colB = st.columns(2)

    with colA:
        if not plan["recoverable"]:
            st.error("Recovery: Not feasible with current task durations.")
        else:
            st.success("Recovery: Feasible by accelerating selected tasks.")

        st.write(f"**Current slip:** `{slip:.1f}` days")

    with colB:
        gate = plan["gatekeepers"]
        if gate is not None and not gate.empty:
            unique_owners = gate["Owner"].fillna("Unassigned").nunique()
            st.write(f"**Owners with leverage:** `{unique_owners}`")

            by_owner_gate = (
                gate.groupby("Owner", dropna=False)["Remaining_LV"]
                .sum()
                .reset_index()
                .sort_values("Remaining_LV", ascending=False)
            )
            st.write("Remaining critical/near-critical work by owner:")
            st.dataframe(by_owner_gate)
        else:
            st.write("No gatekeeper tasks found (critical / near-critical with remaining work).")

st.divider()

# -----------------------------------------------------------
# Specific Scenarios
# -----------------------------------------------------------

scenario = plan["scenario_type"]

if scenario == "single" and plan["selected_tasks"] is not None:
    st.subheader("Scenario 1: Single-Task Recovery")

    t = plan["selected_tasks"].iloc[0]

    st.markdown(
        f"""
        **If you only touch one task, fix this:**

        - **Task:** `{int(t['TaskID'])} – {t['Name']}`
        - **Owner:** `{t['Owner']}`
        - **Remaining duration:** `{t['Remaining_LV']:.1f}` days  
        - **Required cut to recover full slip:** `{t['RequiredCut']:.1f}` days  
        - **New remaining duration:** `{t['NewRemaining']:.1f}` days  
        """
    )

    # Simple bar to show cut vs remaining
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Current Remaining", "After Acceleration"],
        y=[t["Remaining_LV"], t["NewRemaining"]],
        name="Remaining Duration",
    ))
    fig.update_layout(
        title="Single-Task Acceleration Effect",
        yaxis_title="Days",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

elif scenario == "multi" and plan["selected_tasks"] is not None:
    st.subheader("Scenario 2: Minimal Multi-Task Recovery Set")

    sel = plan["selected_tasks"].copy()
    sel_display = sel[
        [
            "TaskID",
            "Name",
            "Owner",
            "Remaining_LV",
            "RequiredCut",
            "NewRemaining",
            "ImpactFactor",
        ]
    ].sort_values("RequiredCut", ascending=False)

    st.markdown(
        f"**Recommended minimal set of tasks to accelerate** "
        f"to recover approximately `{slip:.1f}` days of slip:"
    )

    st.dataframe(sel_display, use_container_width=True)

    # Bar chart: required cut by owner
    by_owner_cut = (
        sel.groupby("Owner", dropna=False)["RequiredCut"]
        .sum()
        .reset_index()
        .sort_values("RequiredCut", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_cut_owner = px.bar(
            by_owner_cut,
            x="Owner",
            y="RequiredCut",
            title="Acceleration Effort by Owner (Days to Cut)",
        )
        fig_cut_owner.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cut_owner, use_container_width=True)

    with col2:
        # Waterfall: Baseline → Live → After recovery
        bl = bl_finish
        lv = lv_finish
        recovered_finish = lv - slip  # assuming full slip recaptured

        fig_wf = go.Figure(go.Waterfall(
            name="Schedule",
            orientation="v",
            measure=["absolute", "relative", "relative"],
            x=["Baseline", "Slip", "Recovery"],
            text=[f"{bl:.1f}", f"{slip:.1f}", f"-{slip:.1f}"],
            y=[bl, slip, -slip],
        ))
        fig_wf.update_layout(
            title="Baseline vs Live vs Recovered Finish (Days)",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

else:
    st.subheader("No Structured Recovery Scenario Available")
    st.write(
        "The engine could not construct a clean single-task or minimal multi-task "
        "recovery plan. This usually means your critical chain is fragmented, "
        "or there isn't enough remaining duration on the controlling tasks."
    )

st.divider()

# -----------------------------------------------------------
# Gatekeeper Heatmap
# -----------------------------------------------------------

st.subheader("Gatekeepers: Who Actually Controls the Date?")

gate = plan["gatekeepers"]
if gate is not None and not gate.empty:
    gate_view = gate[
        [
            "TaskID",
            "Name",
            "Owner",
            "Remaining_LV",
            "Float_LV",
            "CriticalityWeight",
            "SlippageExposure",
        ]
    ].sort_values("SlippageExposure", ascending=False)

    st.dataframe(gate_view, use_container_width=True)

    fig_heat = px.treemap(
        gate,
        path=["Owner", "Name"],
        values="SlippageExposure",
        color="CriticalityWeight",
        color_continuous_scale="Reds",
        title="Slippage Exposure by Owner / Task",
    )
    fig_heat.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No critical or near-critical tasks with remaining work were found.")