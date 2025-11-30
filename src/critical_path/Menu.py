import os, sys

# Absolute path to this file
THIS_FILE = os.path.abspath(__file__)

# Go up ONE directory:
#   Menu.py → critical_path/ → src/
PROJECT_SRC = os.path.abspath(os.path.join(THIS_FILE, "../"))

# Ensure src folder is on PYTHONPATH
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

import streamlit as st
import pandas as pd
from critical_path.cpm.dual_cpm_csv import process_uploaded_dataframe

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
        # Fix datetimes
        for c in ["Start", "Finish", "Baseline Start", "Baseline Finish"]:
            if c in df_raw.columns:
                df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

        df_clean = process_uploaded_dataframe(df_raw)

        st.session_state["schedule_df"] = df_clean
        st.session_state["schedule_name"] = uploaded.name

        st.success(f"Schedule '{uploaded.name}' loaded successfully!")

    except Exception as e:
        st.error(f"Error loading schedule: {e}")