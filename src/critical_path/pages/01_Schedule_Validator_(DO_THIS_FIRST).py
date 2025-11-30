import sys
import os

# -----------------------------------------------------------
# Make sure "critical_path" package is importable
# -----------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
PROJECT_SRC = os.path.abspath(os.path.join(THIS_FILE, "../../../"))

if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

import streamlit as st
import pandas as pd
import numpy as np

from critical_path.validation.schedule_validator import validate_schedule
from critical_path.cpm.dual_cpm_csv import process_uploaded_dataframe

st.set_page_config(
    page_title="Schedule Validator",
    layout="wide"
)

st.title("🔍 Schedule Validator")
st.caption("Ensure your schedule is structurally sound before running analysis.")

uploaded = st.file_uploader("Upload Schedule CSV for Validation", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin validation.")
    st.stop()

# Read CSV
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Fix datetimes
for c in ["Start", "Finish", "Baseline Start", "Baseline Finish"]:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

# Normalize structure (reuses CPM logic)
try:
    df_clean = process_uploaded_dataframe(df_raw)
except Exception as e:
    st.error(f"Validation cannot proceed: {e}")
    st.stop()

# Run validator
try:
    issues = validate_schedule(df_clean)
except Exception as e:
    st.error(f"Error while running validator: {e}")
    st.stop()

st.success("Validation completed.")

# Split: critical vs. warnings
critical = issues[issues["Severity"] == "CRITICAL"]
warnings = issues[issues["Severity"] == "WARNING"]

# Summary
col1, col2 = st.columns(2)
col1.metric("Critical Issues", len(critical))
col2.metric("Warnings", len(warnings))

st.divider()

# Critical issues
st.subheader("❌ Critical Issues")
if critical.empty:
    st.success("No critical issues found.")
else:
    st.error("These issues may break the schedule calculations.")
    st.dataframe(critical)

# Warnings
st.subheader("⚠️ Warnings (Good to Have Fixes)")
if warnings.empty:
    st.success("No warnings.")
else:
    st.warning("These issues won’t break CPM, but should be cleaned up.")
    st.dataframe(warnings)

st.divider()

# Export button
csv_export = issues.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Full Validation Report (CSV)",
    data=csv_export,
    file_name="schedule_validation_results.csv",
    mime="text/csv",
)