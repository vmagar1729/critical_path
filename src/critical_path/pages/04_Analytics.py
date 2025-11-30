import os, sys

# Absolute directory containing Menu.py
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# The project root: Menu.py → critical_path → src
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Deep Analytics", layout="wide")

st.title("📊 Deep Analytics Dashboard")
st.markdown("Advanced project intelligence: slippage, risk clusters, criticality, and intervention priorities.")

# -------------------------------------------------------------------
# Load schedule from session
# -------------------------------------------------------------------
if "schedule_df" not in st.session_state or st.session_state["schedule_df"] is None:
    st.error("No schedule loaded. Please upload a schedule in the Menu page first.")
    st.stop()

df = st.session_state["schedule_df"]

# Ensure intelligence layer exists
required_cols = ["SlippageExposure", "Remaining_LV", "CriticalityWeight",
                 "Float_LV", "PercentComplete", "ExpectedPct"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing intelligence fields: {missing}. Run through CPM Engine first.")
    st.stop()

# -------------------------------------------------------------------
# 1. SLIPPAGE EXPOSURE TREEMAP (heatmap of risk)
# -------------------------------------------------------------------
df["SlippageExposure_safe"] = df["SlippageExposure"].replace(0, 1e-6)
st.subheader("🔥 Slippage Exposure — Risk Concentration Map")

fig1 = px.treemap(
    df,
    path=[px.Constant("All Tasks"), "WBS", "Name"],
    values="SlippageExposure_safe",
    color="SlippageExposure",
    color_continuous_scale="Reds",
    hover_data=["TaskID", "Remaining_LV", "CriticalityWeight"],
)
fig1.update_layout(margin=dict(t=30, l=0, r=0, b=0))

st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
**Interpretation:**  
Large and darker nodes = where schedule danger is concentrated.  
These are your “risk clusters” — if they slip, the schedule slips.
""")

st.divider()

# -------------------------------------------------------------------
# 2. SLIPPAGE vs. PROGRESS SCATTER
# -------------------------------------------------------------------
st.subheader("📉 Progress vs. Expected Timeline (Reality Check)")

df["Delta"] = df["PercentComplete"] - (df["ExpectedPct"] * 100)

fig2 = px.scatter(
    df,
    x="ExpectedPct",
    y="PercentComplete",
    color="Delta",
    color_continuous_scale="RdBu",
    hover_data=["TaskID", "Name", "Remaining_LV", "Float_LV"],
    size="Remaining_LV",
)
fig2.add_shape(
    type="line",
    x0=0, y0=0,
    x1=100, y1=100,
    line=dict(color="gray", dash="dash")
)
fig2.update_layout(
    xaxis_title="Expected % Complete",
    yaxis_title="Actual % Complete",
    height=500,
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**Interpretation:**  
- Points **below** the diagonal line = *behind schedule*  
- Larger bubbles = more remaining work  
- Redder = deeper behind  
""")

st.divider()

# -------------------------------------------------------------------
# 3. TOP TASK RISKS (Slippage Exposure Ranking)
# -------------------------------------------------------------------
st.subheader("🚨 Highest-Risk Tasks (Top Slippage Exposure)")

top_risk = df.sort_values("SlippageExposure", ascending=False).head(15)

st.dataframe(
    top_risk[[
        "TaskID", "Name", "WBS", "PercentComplete",
        "Remaining_LV", "Float_LV", "CriticalityWeight", "SlippageExposure"
    ]]
)

st.divider()

# -------------------------------------------------------------------
# 4. CRITICALITY DISTRIBUTION (Pie Chart)
# -------------------------------------------------------------------
st.subheader("🎯 Criticality Distribution")

crit_data = pd.DataFrame({
    "Status": ["Critical", "Near Critical", "Safe"],
    "Count": [
        df["IsCritical_LV"].sum(),
        df["IsNearCritical_LV"].sum(),
        len(df) - df["IsCritical_LV"].sum() - df["IsNearCritical_LV"].sum(),
    ]
})

fig3 = px.pie(
    crit_data,
    names="Status",
    values="Count",
    color="Status",
    color_discrete_map={
        "Critical": "red",
        "Near Critical": "orange",
        "Safe": "green"
    }
)
fig3.update_traces(textposition='inside', textinfo='percent+label')

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# -------------------------------------------------------------------
# 5. RECOVERY PRIORITIZATION — THE “FIX FIRST” TASKS
# -------------------------------------------------------------------
st.subheader("🛠️ Highest-Impact Recovery Targets")

priority = df.sort_values("SlippageExposure", ascending=False).head(10)

fig4 = px.bar(
    priority,
    x="Name",
    y="SlippageExposure",
    color="CriticalityWeight",
    color_continuous_scale="RdYlGn_r",
    hover_data=["TaskID", "PercentComplete", "Remaining_LV", "Float_LV"],
)
fig4.update_layout(
    xaxis_title="Task",
    yaxis_title="Recovery Impact Score",
    height=450
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
These are the **10 tasks that give you the most schedule recovery per unit effort**.  
Accelerate these, and the whole project stabilizes fastest.
""")

# -------------------------------------------------------------------
# END OF DASHBOARD
# -------------------------------------------------------------------