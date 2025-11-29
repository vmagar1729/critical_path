import streamlit as st
import pandas as pd
import numpy as np

from dual_cpm_csv import compute_dual_cpm_from_df


# -------------------------
# UI START
# -------------------------

st.set_page_config(layout="wide")
st.title("📊 CPM Dashboard – Baseline, Live, Predictive (Coming Soon)")


uploaded = st.file_uploader("Upload CSV from MS Project or PSPLIB-converted CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()


# -------------------------
# LOAD + COMPUTE CPM
# -------------------------

with st.spinner("Computing CPM..."):
    df_raw = pd.read_csv(uploaded)
    df, bl_path, lv_path = compute_dual_cpm_from_df(df_raw)


leaf = df[df["IsLeaf"]]


# -------------------------
# WBS HIERARCHY VIEW
# -------------------------

st.subheader("📁 WBS Hierarchy")

def indent(name, level):
    return ("    " * (level - 1)) + name if level > 1 else name

tree_df = df.copy()
tree_df["Indented"] = tree_df.apply(lambda r: indent(r["Name"], int(r["OutlineLevel"])), axis=1)

st.dataframe(tree_df[["WBS", "Indented", "IsSummary", "IsLeaf"]])


# -------------------------
# BASELINE & LIVE CRITICAL PATHS
# -------------------------

st.subheader("🔗 Critical Paths")

st.write("**Baseline Critical Path:**", bl_path)
st.write("**Live Critical Path:**", lv_path)


# -------------------------
# PACK A – EXECUTIVE HEALTH PANEL
# -------------------------

st.markdown("## 📈 Executive Health Overview")

leaf = df[df["IsLeaf"]]

behind = leaf[leaf["BehindSchedule"] == True]
num_behind = len(behind)
total_leaf = len(leaf)
pct_behind = (num_behind / total_leaf * 100) if total_leaf else 0

# finish dates
bl_finish = leaf.loc[leaf["BL_EF"].idxmax(), "BL_EF"]
lv_finish = leaf.loc[leaf["LV_EF"].idxmax(), "LV_EF"]
slip_days = (lv_finish - bl_finish).days

k1, k2, k3 = st.columns(3)
k1.metric("Tasks Behind", f"{num_behind}/{total_leaf}", f"{pct_behind:.1f}%")
k2.metric("Baseline Finish", bl_finish)
k3.metric("Live Finish", lv_finish, f"{slip_days:+d} days")


# Expected vs Actual chart
st.markdown("### Expected vs Actual % Complete")
started = leaf[leaf["ExpectedPercent"].notna()]

if not started.empty:
    comp = started.sort_values("ExpectedPercent", ascending=False).head(25)
    comp["Expected"] = comp["ExpectedPercent"]
    comp["Actual"] = comp["PercentComplete"]

    st.bar_chart(comp.set_index("Name")[["Expected", "Actual"]])
else:
    st.info("No tasks have baseline expectations yet.")


# WBS Slip Index
st.markdown("### Workstream Slip Index")

wbs_groups = (
    leaf.groupby("WBS")
    .apply(lambda g: (g["PercentComplete"] - g["ExpectedPercent"]).mean(skipna=True))
    .reset_index(name="SlipIndex")
).sort_values("SlipIndex")

st.dataframe(wbs_groups)


# Critical behind
st.markdown("### Critical Tasks Behind Schedule")

crit_behind = leaf[
    (leaf["LV_Critical"] == True) &
    (leaf["BehindSchedule"] == True)
]

if crit_behind.empty:
    st.success("No critical tasks are behind.")
else:
    st.dataframe(
        crit_behind[
            ["WBS", "Name", "ExpectedPercent", "PercentComplete", "Remaining_LV", "LV_Float"]
        ]
    )


# -------------------------
# FULL DATA TABLE
# -------------------------

st.subheader("📑 Full Task Table")
st.dataframe(df)