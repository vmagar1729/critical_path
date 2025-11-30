import streamlit as st
import pandas as pd

from critical_path.validation.schedule_validator import validate_schedule


st.set_page_config(page_title="Schedule Validator", layout="wide")
st.title("🩺 Schedule Validator (DO THIS FIRST)")


# -----------------------------------------------------
# Require the user to upload schedule in Menu
# -----------------------------------------------------
if "schedule_df" not in st.session_state or st.session_state["schedule_df"] is None:
    st.warning("⚠️ Please upload a schedule using the **Menu** page first.")
    st.stop()

df = st.session_state["schedule_df"].copy()
schedule_name = st.session_state.get("schedule_name", "(Unnamed Schedule)")

st.markdown(f"### Validating: **{schedule_name}**")


# -----------------------------------------------------
# Run validator engine
# -----------------------------------------------------
issues = validate_schedule(df)

if not issues:
    st.success("🎉 No scheduling issues detected. Your plan is frighteningly clean!")
    st.stop()

df_issues = pd.DataFrame(issues)


# -----------------------------------------------------
# Group by severity
# -----------------------------------------------------
severity_order = ["critical", "error", "warning", "minor"]

st.markdown("## 🧹 Validation Results")

for sev in severity_order:
    subset = df_issues[df_issues["Severity"] == sev]
    if subset.empty:
        continue

    header = {
        "critical": "🔴 **Critical Issues** — must be fixed or CPM fails",
        "error": "🟠 **Errors** — will cause downstream problems",
        "warning": "🟡 **Warnings** — recommended cleanup",
        "minor": "🟢 **Minor Notes** — optional polish",
    }.get(sev, sev)

    st.markdown(f"### {header} ({len(subset)})")

    st.dataframe(
        subset[
            ["TaskID", "Name", "IssueType", "Description", "SuggestedFix"]
        ].sort_values("TaskID")
    )

    st.divider()

st.info("Fix issues and re-upload the schedule before running CPM.")