# critical_path/pages/05_FAQ_&_Glossary.py

import os
import sys

# -------------------------------------------------------------------
# Path bootstrap: same pattern as other pages
# -------------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FAQ & Glossary",
    layout="wide",
)

st.title("📘 FAQ & Glossary")
st.caption("Quick reference for the terms and metrics used across the dashboards.")

st.markdown("---")

# ===================================================================
# FAQ SECTION
# ===================================================================
st.header("❓ Frequently Asked Questions")

with st.expander("1. What does this platform actually do?", expanded=True):
    st.markdown(
        """
This engine ingests your schedule data, rebuilds baseline and live critical paths,
and computes a set of metrics that answer four core questions:

1. **Are we on track?**  
2. **If not, how far behind are we?**  
3. **Who is contributing most to the delay?**  
4. **What would we need to change to recover the date?**  

It does this using Critical Path Method (CPM), float analysis, duration creep detection,
owner-level aggregation, and (optionally) recovery / prediction logic.
        """
    )

with st.expander("2. Why are some dates or metrics showing as N/A?"):
    st.markdown(
        """
Most often this happens when:

- Date fields (Start, Finish, Baseline Start, Baseline Finish) are blank or malformed  
- Baseline was never set in Microsoft Project  
- The CSV export omitted required fields  
- All dates for a column failed to parse into valid timestamps  

Fix: ensure all required columns are exported and that dates look like real dates,
not text placeholders.
        """
    )

with st.expander("3. Why does the tool say tasks are behind when people say they are fine?"):
    st.markdown(
        """
Behind/ahead status is computed mathematically, not subjectively.

A task is considered **behind** if:

- Its **actual % complete (`PercentComplete`)** is less than  
- Its **expected % complete (`ExpectedPct`)** based on the baseline dates

and/or its **live early finish (`LV_EF`)** is later than **baseline early finish (`BL_EF`)**.

If those don’t align with verbal status, the schedule and the narrative are out of sync.
        """
    )

with st.expander("4. What is “Slippage Exposure” actually measuring?"):
    st.markdown(
        """
**Slippage Exposure** estimates how much a task can impact the project finish date
if its remaining work slips further.

It combines:

- Remaining live duration  
- Whether the task is critical or near-critical  
- How close it is to consuming all available float  

Higher exposure = greater potential to move the project finish.
        """
    )

with st.expander("5. What is “Float” and why is it important?"):
    st.markdown(
        """
**Float** (or slack) is how many days a task can slip before it delays the project finish.

- **Zero float** → the task is on the **critical path**  
- **Positive float** → the task has some buffer  
- **Negative float** → the current target date is earlier than what the network logic supports  

Float is what determines which tasks actually control the end date.
        """
    )

with st.expander("6. Why does the engine say the schedule is unrecoverable?"):
    st.markdown(
        """
The recovery engine looks at:

- Current slip between baseline finish and live finish  
- Remaining work on critical / near-critical tasks  
- Available float on the network  

If the total remaining duration on the controlling tasks is **less than** the required
recovery time, then even aggressive acceleration on those tasks cannot fully recover
the slip. In that case, recovery requires:

- Changing scope  
- Moving the target date  
- Re-sequencing major portions of the plan
        """
    )

with st.expander("7. Why are Owners highlighted so prominently in charts?"):
    st.markdown(
        """
Tasks are aggregated by **Owner** to show:

- How much of the delay sits with each owner/team  
- Where the remaining work is concentrated  
- Who actually has leverage over the end date  

This is not about blame; it is about identifying where intervention has the most impact.
        """
    )

with st.expander("8. How does the recovery engine pick which tasks to accelerate?"):
    st.markdown(
        """
The recovery logic focuses on **gatekeeper tasks**:

- Tasks with low or zero float (critical / near-critical)  
- Tasks with non-zero remaining duration  
- Tasks with higher criticality weight and exposure  

It then constructs either:

- A **single-task scenario** (one task can recover the full slip), or  
- A **minimal multi-task scenario** (smallest set of tasks whose combined acceleration
  can recover the slip)

This is based on schedule logic, not org charts.
        """
    )

with st.expander("9. What does the Monte-Carlo simulation do (when enabled)?"):
    st.markdown(
        """
Monte-Carlo simulation runs many randomized scenarios by perturbing task durations
within a configured range.

The engine then:

- Computes resulting finish dates for each scenario  
- Builds a probability distribution of completion dates  
- Reports **likelihood of hitting the current target date**  

It answers: *“Given current variability, how realistic is the date we’re promising?”*
        """
    )

with st.expander("10. Why do some tasks show negative float?"):
    st.markdown(
        """
**Negative float** means the required finish date is earlier than the date implied
by the current network logic and durations.

In other words, the plan is already mathematically late relative to the imposed deadline.
        """
    )

with st.expander("11. What is the Executive Narrative feature?"):
    st.markdown(
        """
The **Executive Narrative** is a natural-language summary built from:

- Current slip and risk profile  
- Top owners and tasks driving delay  
- Recovery options (if any)  
- Trend signals (float consumption, duration creep)  

It is intended as a concise “board-ready” explanation of where the project stands and why.
        """
    )

st.markdown("---")

# ===================================================================
# GLOSSARY SECTION
# ===================================================================
st.header("📚 Glossary of Key Terms")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Schedule & CPM")

    st.markdown(
        """
**Baseline**  
The original approved schedule used as the reference point for measuring performance.

**Live Schedule**  
The current working schedule after updates, re-planning, and progress entries.

**Critical Path**  
The longest chain of dependent tasks in terms of duration. Any delay on this path
directly delays the project finish.

**Near-Critical Path**  
Tasks with low positive float that can quickly become critical if they slip.

**Float / Slack**  
The amount of time a task can be delayed without delaying the project finish date.
Zero float = on critical path. Negative float = impossible target relative to current logic.

**Early Start / Early Finish (ES / EF)**  
The earliest possible start and finish dates for a task given dependencies.

**Late Start / Late Finish (LS / LF)**  
The latest allowable start and finish dates without delaying the project finish.

**Total Float**  
`LS - ES` (or equivalently `LF - EF`), i.e., the flexibility on a task before it affects the end date.
        """
    )

with col2:
    st.subheader("Performance & Risk")

    st.markdown(
        """
**PercentComplete**  
The actual reported completion percentage of a task (0–100%).

**ExpectedPct**  
The planned percent complete for a task by “today”, based on its baseline dates.

**Schedule Variance (SV%)**  
The difference between actual and expected percent complete.  
Negative = behind schedule; positive = ahead.

**Remaining_LV (Remaining Duration)**  
Estimated remaining duration for a task, based on live duration and current completion %.

**Slippage Exposure**  
A risk-weighted measure of how much a task can affect the project finish if it continues to slip.
Combines remaining duration, float, and criticality.

**CriticalityWeight**  
A weight reflecting how strongly a task controls the finish date (e.g., 1.0 for critical, lower for near/non-critical).

**Duration Creep**  
The increase in duration from baseline to live (e.g., baseline 5 days → live 8 days = 3 days of creep).

**Owner**  
The person or team responsible for a task.

**Gatekeeper Task**  
A task with both meaningful remaining work and low/zero float, making it structurally able to move the finish date.

**ImpactFactor**  
Composite metric integrating Remaining_LV, CriticalityWeight, and exposure, used to rank tasks by influence on schedule risk.

**Slack Burndown**  
A trend of how total available float is being consumed over time. Downward trends indicate tightening schedule flexibility.

**Monte-Carlo Simulation**  
A risk analysis technique where task durations are randomly varied many times to estimate the distribution of possible completion dates.

**Risk Cluster**  
A group of tasks with similar risk behavior (e.g., high creep, low float, same owner or work type), often identified algorithmically.

**Executive Narrative**  
An automatically generated summary explaining the schedule state, key risks, and recovery options in business language.
        """
    )

st.markdown("---")

st.info(
    "Use this page as the reference point whenever terms like *float*, "
    "*slippage exposure*, or *gatekeeper tasks* appear on other dashboards."
)