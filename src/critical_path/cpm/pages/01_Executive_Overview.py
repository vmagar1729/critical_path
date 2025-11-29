import streamlit as st
import pandas as pd
import numpy as np


def render_executive_overview(df):
    st.title("📊 Executive Overview")

    # ----------------------------
    # Project-level aggregations
    # ----------------------------
    dur = df["Dur_BL"]

    project_expected = (dur * df["ExpectedPct"]).sum() / dur.sum()
    project_actual   = (dur * df["PercentComplete"]).sum() / dur.sum()
    project_variance = project_actual - project_expected

    baseline_finish = df["BL_EF"].max()
    live_finish     = df["LV_EF"].max()
    forecast_delay  = live_finish - baseline_finish

    num_late_tasks    = (df["ScheduleVariance"] < 0).sum()
    num_critical_late = ((df["IsCritical_LV"]) & (df["ScheduleVariance"] < 0)).sum()

    risk_score = df["SlippageExposure"].sum()

    # ----------------------------
    # KPI Grid
    # ----------------------------
    c1, c2, c3 = st.columns(3)
    c4, c5 = st.columns(2)

    c1.metric("Actual % Complete", f"{project_actual*100:.1f}%")
    c2.metric("Expected % Complete", f"{project_expected*100:.1f}%")
    c3.metric(
        "Variance",
        f"{project_variance*100:.1f}%",
        delta=f"{project_variance*100:.1f}%",
        delta_color="inverse"   # negative should be red
    )

    c4.metric("Forecast Finish", f"{live_finish:.1f} days")
    c5.metric("Delay vs Baseline", f"{forecast_delay:.1f} days",
              delta=f"{forecast_delay:.1f}",
              delta_color="inverse")

    st.markdown("---")

    # ----------------------------
    # Health Indicators
    # ----------------------------
    st.subheader("Health Indicators")

    h1, h2, h3 = st.columns(3)
    h1.metric("Late Tasks", num_late_tasks)
    h2.metric("Critical Late Tasks", num_critical_late)
    h3.metric("Exposure Score", f"{risk_score:.1f}")

    st.markdown("---")

    # ----------------------------
    # Interpretation Block
    # ----------------------------
    st.subheader("Interpretation")

    bullet = []

    if project_variance < -0.05:
        bullet.append("• The project is behind schedule in aggregate progress.")
    elif project_variance > 0.05:
        bullet.append("• The project is ahead of schedule.")
    else:
        bullet.append("• The project is tracking close to the expected plan.")

    if forecast_delay > 1:
        bullet.append(f"• Forecasted completion is **{forecast_delay:.1f} days** later than baseline.")
    elif forecast_delay < -1:
        bullet.append(f"• Forecasted completion is **{abs(forecast_delay):.1f} days earlier** than baseline.")
    else:
        bullet.append("• Forecasted completion is approximately on baseline.")

    if num_critical_late > 0:
        bullet.append("• Critical-path slippage detected. Immediate intervention recommended.")
    else:
        bullet.append("• No critical-path tasks are currently behind schedule.")

    if risk_score > 15:
        bullet.append("• Overall project risk is high due to clustered slippage on important tasks.")
    else:
        bullet.append("• Project risk remains manageable based on current exposure.")

    st.write("\n".join(bullet))

