import os, sys
import streamlit as st
import pandas as pd

# Absolute directory containing Menu.py
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# The project root: Menu.py → critical_path → src
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ✅ IMPORT THE FULL ENGINE
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

# --- UI LOGIC: Use a placeholder to manage the uploader visibility ---
uploader_placeholder = st.empty()

# Check if we already have a file loaded
if st.session_state["schedule_df"] is not None:
    # File is loaded; show success and a reset button
    with uploader_placeholder.container():
        st.success(f"✅ Active Schedule: **{st.session_state['schedule_name']}**")
        st.info("Navigate to other pages (Dashboard, Gantt, etc.) to analyze this schedule.")

        if st.button("🔄 Upload Different Schedule"):
            # Clear state to bring back the uploader
            st.session_state["schedule_df"] = None
            st.session_state["schedule_name"] = None
            st.rerun()

else:
    # No file loaded; show the uploader inside the placeholder
    with uploader_placeholder.container():
        uploaded = st.file_uploader("Upload Schedule CSV", type=["csv"])

        if uploaded:
            try:
                # Read CSV
                df_raw = pd.read_csv(uploaded)

                # NOTE: Date conversion removed here because
                # compute_dual_cpm_from_df -> process_uploaded_dataframe handles it internally.

                # 🔥 Run full CPM + intelligence pipeline
                with st.spinner("Running Critical Path Analysis..."):
                    df_cpm = compute_dual_cpm_from_df(df_raw)

                # Store enriched DF for all pages
                st.session_state["schedule_df"] = df_cpm
                st.session_state["schedule_name"] = uploaded.name

                # Force rerun to hide uploader and show success state
                st.rerun()

            except ValueError as ve:
                # Specific handling for Engine errors (like Circular Dependencies)
                st.error(f"❌ Scheduling Logic Error: {ve}")
                if "acyclic" in str(ve).lower():
                    st.warning(
                        "💡 Hint: This usually means your schedule has a loop (A->B->A). Check your predecessors.")

            except Exception as e:
                # Generic fallback for other errors (parsing, missing columns)
                st.error(f"❌ Error loading schedule: {e}")