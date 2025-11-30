import pandas as pd
import numpy as np
from collections import defaultdict, deque
import re

# ---------------------------------------------------------
# FIELD CLEANUP & PREPARATION
# ---------------------------------------------------------

def process_uploaded_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a DataFrame exported from MS Project
    OR generated from PSPLIB loader.

    Guarantees:
      - TaskID is numeric
      - Duration numeric (Dur_BL created)
      - PercentComplete numeric (0–100)
      - Date fields are datetime (or NaT)
      - Predecessors/WBS/Outline Level exist or are safely defaulted
    """
    df = df_input.copy()

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # ---- TaskID ----
    if "TaskID" not in df.columns:
        raise ValueError("Missing required column: 'TaskID'")
    df["TaskID"] = pd.to_numeric(df["TaskID"], errors="coerce")

    if df["TaskID"].isna().any():
        bad = df[df["TaskID"].isna()][["TaskID", "Name"]].head()
        raise ValueError(
            "Non-numeric TaskID values found. Example rows:\n"
            f"{bad.to_string(index=False)}"
        )
    df["TaskID"] = df["TaskID"].astype(int)

    # ---- Duration / Dur_BL ----
    if "Duration" not in df.columns:
        raise ValueError("Duration column missing — cannot compute CPM.")
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    df["Dur_BL"] = df["Duration"]

    # ---- Percent Complete → PercentComplete (0–100) ----
    if "% Complete" in df.columns:
        pct = (
            df["% Complete"]
            .astype(str)
            .str.replace("%", "", regex=False)
        )
        df["PercentComplete"] = pd.to_numeric(pct, errors="coerce").fillna(0.0)
    elif "PercentComplete" in df.columns:
        df["PercentComplete"] = pd.to_numeric(df["PercentComplete"], errors="coerce").fillna(0.0)
    else:
        df["PercentComplete"] = 0.0

    # Clamp to sane range
    df["PercentComplete"] = df["PercentComplete"].clip(lower=0.0, upper=100.0)

    # ---- Dates ----
    date_cols = ["Start", "Finish", "Baseline Start", "Baseline Finish"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            # Ensure column exists as NaT if missing
            df[col] = pd.NaT

    # ---- Predecessors (optional, but needed for parsing) ----
    if "Predecessors" not in df.columns:
        df["Predecessors"] = ""

    # ---- WBS / Outline Level (optional) ----
    if "WBS" not in df.columns:
        # Fallback: trivial WBS, one level per task
        df["WBS"] = df["TaskID"].astype(int).astype(str)

    if "Outline Level" not in df.columns:
        # If you ever need it later, at least it's there
        df["Outline Level"] = 1

    # ---- Name ----
    if "Name" not in df.columns:
        raise ValueError("Missing required column: 'Name'")

    # Final minimal required set for CPM
    required = [
        "TaskID", "Name",
        "Duration", "Dur_BL",
        "Baseline Start", "Baseline Finish",
        "Predecessors",
        "WBS",
        "PercentComplete",
        "Owner"
    ]
    df["Owner"] = df["Owner"].fillna("Unassigned").astype(str)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    return df

# ---------------------------------------------------------
# WBS HIERARCHY
# ---------------------------------------------------------

def build_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "WBS" not in df.columns:
        df["WBS"] = df["TaskID"].astype(int).astype(str)

    wbs_series = df["WBS"].astype(str)

    # Summary task = has children whose WBS starts with this WBS + "."
    is_summary = []
    for w in wbs_series:
        prefix = w + "."
        has_child = any((x != w) and x.startswith(prefix) for x in wbs_series)
        is_summary.append(has_child)

    df["IsSummary"] = is_summary
    df["IsLeaf"] = ~df["IsSummary"]

    return df

# ---------------------------------------------------------
# PREDECESSOR PARSING
# ---------------------------------------------------------

def parse_predecessor_cell(cell):
    """
    Parse a Predecessors cell like:
      "5"
      "5FS+3d"
      "12SS-2"
      "7FF+1d, 9SS"
    into a list of tuples:
      [(5, "FS", 0.0), (12, "SS", -2.0), ...]
    """

    if cell is None:
        return []

    text = str(cell).strip()
    if not text:
        return []

    parts = re.split(r"[;,]", text)

    results = []
    pattern = re.compile(
        r"""
        ^\s*
        (?P<pred>\d+)
        \s*
        (?P<type>FS|SS|FF|SF)?    # optional type
        \s*
        (?P<lag>[+-]\s*\d+)?      # optional +N or -N
        \s*[dD]?                  # optional 'd'
        \s*$
        """,
        re.VERBOSE,
    )

    for raw in parts:
        s = raw.strip()
        if not s:
            continue
        m = pattern.match(s)
        if not m:
            # Fallback: if it's just a number, treat as FS+0
            if s.isdigit():
                results.append((int(s), "FS", 0.0))
            continue

        pred = int(m.group("pred"))
        dep_type = m.group("type") or "FS"
        lag_str = m.group("lag")
        if lag_str:
            lag = float(lag_str.replace(" ", ""))
        else:
            lag = 0.0

        results.append((pred, dep_type, lag))

    return results

# ---------------------------------------------------------
# GRAPH CONSTRUCTION (Baseline or Live)
# ---------------------------------------------------------

def build_graph(df: pd.DataFrame, mode: str = "baseline"):
    """
    Build a dependency-aware DAG:

    nodes:      sorted list of task IDs
    durations:  dict {task_id: duration_in_days}
    edges_from: dict {pred: [(succ, dep_type, lag), ...]}
    edges_to:   dict {succ: [(pred, dep_type, lag), ...]}

    mode:
      baseline → use Dur_BL
      live     → use Duration (fallback to Dur_BL when missing/0)
    """

    # 1. Select duration source
    if mode == "baseline":
        dur_series = df["Dur_BL"]
    elif mode == "live":
        dur_series = (
            df["Duration"]
            .replace(0, np.nan)
            .fillna(df["Dur_BL"])
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Assemble duration dict
    durations = {}
    for idx, row in df.iterrows():
        tid = int(row["TaskID"])
        d = float(dur_series.loc[idx])
        durations[tid] = max(d, 0.0)

    nodes = sorted(durations.keys())

    # 3. Dependencies
    edges_from = defaultdict(list)
    edges_to = defaultdict(list)

    for _, row in df.iterrows():
        succ = int(row["TaskID"])
        preds = parse_predecessor_cell(row["Predecessors"])

        for pred, dep_type, lag in preds:
            if pred not in durations:
                continue
            edges_from[pred].append((succ, dep_type, lag))
            edges_to[succ].append((pred, dep_type, lag))

    return nodes, durations, edges_from, edges_to

# ---------------------------------------------------------
# CPM ALGORITHM
# ---------------------------------------------------------

def compute_cpm(nodes, durations, edges_from, edges_to):
    """
    Compute ES/EF/LS/LF/Float on a DAG with typed, lagged dependencies.

    Returns:
      es, ef, ls, lf, total_float
      each is a dict keyed by task ID
    """

    # Topological order
    indeg = {n: 0 for n in nodes}
    for succ, preds in edges_to.items():
        indeg[succ] = len(preds)

    q = deque([n for n in nodes if indeg[n] == 0])
    topo = []
    while q:
        n = q.popleft()
        topo.append(n)
        for succ, _, _ in edges_from.get(n, []):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                q.append(succ)

    if len(topo) != len(nodes):
        raise ValueError("Graph is not acyclic; cannot compute CPM.")

    # Forward pass
    es = {n: 0.0 for n in nodes}
    ef = {n: 0.0 for n in nodes}
    max_ef_constraint = {n: 0.0 for n in nodes}

    for n in topo:
        d = durations[n]
        ef[n] = max(es[n] + d, max_ef_constraint[n])

        for succ, dep_type, lag in edges_from.get(n, []):
            if dep_type == "FS":
                cand_es = ef[n] + lag
                es[succ] = max(es.get(succ, 0.0), cand_es)
            elif dep_type == "SS":
                cand_es = es[n] + lag
                es[succ] = max(es.get(succ, 0.0), cand_es)
            elif dep_type == "FF":
                cand_ef = ef[n] + lag
                max_ef_constraint[succ] = max(max_ef_constraint[succ], cand_ef)
            elif dep_type == "SF":
                cand_ef = es[n] + lag
                max_ef_constraint[succ] = max(max_ef_constraint[succ], cand_ef)
            else:
                cand_es = ef[n] + lag
                es[succ] = max(es.get(succ, 0.0), cand_es)

    # Finalize EF
    for n in nodes:
        d = durations[n]
        ef[n] = max(es[n] + d, max_ef_constraint[n])

    project_finish = max(ef.values()) if ef else 0.0

    # Backward pass
    lf = {n: project_finish for n in nodes}
    ls = {n: project_finish - durations[n] for n in nodes}

    for n in reversed(topo):
        d = durations[n]
        for succ, dep_type, lag in edges_from.get(n, []):
            if dep_type == "FS":
                lf[n] = min(lf[n], ls[succ] - lag)
            elif dep_type == "SS":
                ls[n] = min(ls[n], ls[succ] - lag)
            elif dep_type == "FF":
                lf[n] = min(lf[n], lf[succ] - lag)
            elif dep_type == "SF":
                ls[n] = min(ls[n], lf[succ] - lag)
            else:
                lf[n] = min(lf[n], ls[succ] - lag)

        ls[n] = min(ls[n], lf[n] - d)

    total_float = {n: ls[n] - es[n] for n in nodes}
    return es, ef, ls, lf, total_float

# ---------------------------------------------------------
# COMPILE RESULTS
# ---------------------------------------------------------

def compile_results(df: pd.DataFrame, bl_data, lv_data):
    ES_bl, EF_bl, LS_bl, LF_bl, Float_bl, CP_bl = bl_data
    ES_lv, EF_lv, LS_lv, LF_lv, Float_lv, CP_lv = lv_data

    df2 = df.copy()

    # Baseline
    df2["BL_ES"] = df2["TaskID"].map(ES_bl)
    df2["BL_EF"] = df2["TaskID"].map(EF_bl)
    df2["BL_LS"] = df2["TaskID"].map(LS_bl)
    df2["BL_LF"] = df2["TaskID"].map(LF_bl)
    df2["BL_Float"] = df2["TaskID"].map(Float_bl)
    df2["BL_Critical"] = df2["TaskID"].isin(CP_bl)
    df2["Float_BL"] = df2["BL_LS"] - df2["BL_ES"]

    # Live
    df2["LV_ES"] = df2["TaskID"].map(ES_lv)
    df2["LV_EF"] = df2["TaskID"].map(EF_lv)
    df2["LV_LS"] = df2["TaskID"].map(LS_lv)
    df2["LV_LF"] = df2["TaskID"].map(LF_lv)
    df2["LV_Float"] = df2["TaskID"].map(Float_lv)
    df2["LV_Critical"] = df2["TaskID"].isin(CP_lv)
    df2["Float_LV"] = df2["LV_LS"] - df2["LV_ES"]

    # ExpectedPercent (0–100) for legacy use
    today = pd.Timestamp.today().normalize()
    dur = df2["Dur_BL"].replace(0, np.nan)
    elapsed_days = (today - df2["Baseline Start"]).dt.days.clip(lower=0)
    df2["ExpectedPercent"] = (elapsed_days / dur * 100).clip(0, 100).fillna(0)

    # Simple flag
    df2["BehindSchedule"] = df2["PercentComplete"] < df2["ExpectedPercent"]

    # Hierarchy fields
    df2 = build_hierarchy(df2)

    return df2, CP_bl, CP_lv

# ---------------------------------------------------------
# INTELLIGENCE LAYER
# ---------------------------------------------------------

def add_intelligence_metrics(df: pd.DataFrame, today=None, near_crit_threshold: float = 1.0) -> pd.DataFrame:
    """
    Adds intelligence-layer analytics needed by dashboards:
      - ExpectedPct (0–1)
      - ScheduleVariance (fraction)
      - Critical / near-critical flags
      - CriticalityWeight
      - Remaining_LV (live remaining duration)
      - SlippageExposure
      - InterventionValue
    """
    df = df.copy()

    if today is None:
        today = pd.Timestamp.today().normalize()

    # 1. Expected % complete (0–1)
    dur = df["Dur_BL"].replace(0, np.nan)
    elapsed = (today - df["Baseline Start"]).dt.days.clip(lower=0)
    expected_pct = (elapsed / dur).clip(lower=0, upper=1).fillna(0)
    df["ExpectedPct"] = expected_pct

    # 2. Schedule variance (fractions)
    df["ScheduleVariance"] = (df["PercentComplete"] / 100.0) - df["ExpectedPct"]

    # 3. Criticality flags
    def near_zero(x, eps=1e-6):
        return pd.notnull(x) and abs(x) < eps

    df["IsCritical_BL"] = df["Float_BL"].apply(near_zero)
    df["IsCritical_LV"] = df["Float_LV"].apply(near_zero)
    df["BecameCritical"] = (~df["IsCritical_BL"]) & df["IsCritical_LV"]

    df["IsNearCritical_LV"] = df["Float_LV"].apply(
        lambda x: pd.notnull(x) and (x > 0) and (x <= near_crit_threshold)
    )

    # 4. Criticality weight
    def crit_weight(row):
        if row["IsCritical_LV"]:
            return 1.0
        if row["IsNearCritical_LV"]:
            return 0.6
        return 0.1

    df["CriticalityWeight"] = df.apply(crit_weight, axis=1)

    # 5. Remaining_LV (canonical)
    live_dur = (
        df["Duration"]
        .replace(0, np.nan)
        .fillna(df["Dur_BL"])
    )
    pct = (df["PercentComplete"] / 100.0).clip(lower=0.0, upper=1.0)
    df["Remaining_LV"] = (1.0 - pct) * live_dur

    # 6. Slippage exposure
    df["SlippageExposure"] = df["Remaining_LV"] * df["CriticalityWeight"]

    # 7. Intervention value
    df["InterventionValue"] = (
        df["IsCritical_LV"] & (df["PercentComplete"] < 100.0)
    ).astype(int)

    return df

# ---------------------------------------------------------
# EXPORTED ENTRY POINT
# ---------------------------------------------------------

def compute_dual_cpm_from_df(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline:
      1. Normalize DF structure
      2. Compute baseline CPM
      3. Compute live CPM
      4. Merge results
      5. Add intelligence metrics

    Returns a single DataFrame with all fields.
    """
    # 1. Normalize
    df = process_uploaded_dataframe(df_input)

    # 2. Baseline CPM
    nodes_bl, durations_bl, edges_from_bl, edges_to_bl = build_graph(df, mode="baseline")
    es_bl, ef_bl, ls_bl, lf_bl, tf_bl = compute_cpm(nodes_bl, durations_bl, edges_from_bl, edges_to_bl)
    bl_cp = [n for n in nodes_bl if abs(tf_bl[n]) < 1e-6]
    bl_data = (es_bl, ef_bl, ls_bl, lf_bl, tf_bl, bl_cp)

    # 3. Live CPM
    nodes_lv, durations_lv, edges_from_lv, edges_to_lv = build_graph(df, mode="live")
    es_lv, ef_lv, ls_lv, lf_lv, tf_lv = compute_cpm(nodes_lv, durations_lv, edges_from_lv, edges_to_lv)
    lv_cp = [n for n in nodes_lv if abs(tf_lv[n]) < 1e-6]
    lv_data = (es_lv, ef_lv, ls_lv, lf_lv, tf_lv, lv_cp)

    # 4. Merge
    df_out, _, _ = compile_results(df, bl_data, lv_data)

    # 5. Intelligence
    df_out = add_intelligence_metrics(df_out)

    return df_out