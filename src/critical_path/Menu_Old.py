import os, sys

# Absolute directory containing Menu_Old.py
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# The project root: Menu_Old.py → critical_path → src
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
# ✅ IMPORT THE FULL ENGINE, NOT JUST THE CLEANER
from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df

st.set_page_config(page_title="Project Intelligence Engine", layout="wide")

st.title("🚀 Project Intelligence Engine")

st.markdown("""
Upload your schedule here once — all pages will use it automatically.
""")

# Initialize session storage
if "schedule_df" not in st.session_state:
    st.session_state["schedule_df"] = None
if "schedule_name" not in st.session_state:
    st.session_state["schedule_name"] = None

uploaded = st.file_uploader("Upload Schedule CSV", type=["csv"])

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)

        # Fix datetimes BEFORE feeding into engine
        for c in ["Start", "Finish", "Baseline Start", "Baseline Finish"]:
            if c in df_raw.columns:
                df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

        # 🔥 Run full CPM + intelligence pipeline
        df_cpm = compute_dual_cpm_from_df(df_raw)

        # Store enriched DF for all pages
        st.session_state["schedule_df"] = df_cpm
        st.session_state["schedule_name"] = uploaded.name

        st.success(f"Schedule '{uploaded.name}' loaded successfully!")

    except Exception as e:
        st.error(f"Error loading schedule: {e}")