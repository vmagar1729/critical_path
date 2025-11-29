import pandas as pd
import numpy as np
from collections import defaultdict, deque
import datetime
import re


# ---------------------------------------------------------
# FIELD CLEANUP & PREPARATION
# ---------------------------------------------------------

def process_uploaded_dataframe(df_input):
    """
    Clean and normalize a DataFrame exported from MS Project
    OR generated from PSPLIB loader.
    Duration is preserved exactly as supplied.
    """

    df = df_input.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalize Duration → Dur_BL
    if "Duration" in df.columns:
        df["Dur_BL"] = pd.to_numeric(df["Duration"], errors="coerce")
    else:
        raise ValueError("Duration column missing — cannot compute CPM.")

    # Normalize Percent Complete
    if "% Complete" in df.columns:
        df["PercentComplete"] = (
            df["% Complete"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )
    elif "PercentComplete" in df.columns:
        df["PercentComplete"] = df["PercentComplete"].astype(float)
    else:
        df["PercentComplete"] = 0.0

    required = [
        "TaskID", "Name",
        "Start", "Finish",
        "Baseline Start", "Baseline Finish",
        "Predecessors",
        "WBS", "Outline Level",
        "% Complete", "PercentComplete",
        "Duration", "Dur_BL"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ⭐ DO NOT drop other columns — keep full DF
    return df

# ---------------------------------------------------------
# WBS HIERARCHY
# ---------------------------------------------------------

def build_hierarchy(df):
    df = df.copy()

    # Summary = has children
    df["IsSummary"] = df["WBS"].apply(lambda w: any(
        w != x and x.startswith(w + ".") for x in df["WBS"]
    ))
    df["IsLeaf"] = ~df["IsSummary"]

    return df

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

def build_graph(df, mode="baseline"):
    """
    Build a dependency-aware DAG:

    nodes: set of task IDs
    durations: dict {task_id: duration_in_days}
    edges_from: dict {pred: [(succ, dep_type, lag), ...]}
    edges_to:   dict {succ: [(pred, dep_type, lag), ...]}

    mode:
      "baseline" → use Dur_BL
      "live"     → use Remaining_LV (fallback to Dur_BL if missing)
    """

    # Duration source
    if mode == "baseline":
        dur_series = df["Dur_BL"]
    elif mode == "live":
        if "Remaining_LV" in df.columns:
            dur_series = df["Remaining_LV"].fillna(df["Dur_BL"])
        else:
            dur_series = df["Dur_BL"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    durations = {}
    for _, row in df.iterrows():
        tid = int(row["TaskID"])
        d = float(dur_series.loc[row.name])
        if d < 0:
            d = 0.0
        durations[tid] = d

    nodes = sorted(durations.keys())

    edges_from = defaultdict(list)
    edges_to = defaultdict(list)

    for _, row in df.iterrows():
        succ = int(row["TaskID"])
        preds = parse_predecessor_cell(row["Predecessors"])
        for pred, dep_type, lag in preds:
            if pred not in durations:
                # ignore invalid preds here; your validator will catch them
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

    nodes: list of task IDs
    durations: {task: duration}
    edges_from: {pred: [(succ, type, lag), ...]}
    edges_to:   {succ: [(pred, type, lag), ...]}

    Returns:
      es, ef, ls, lf, total_float
      each is a dict keyed by task ID
    """

    # -----------------------------
    # Topological order
    # -----------------------------
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
        # Cycle somewhere; your validator should catch that earlier
        raise ValueError("Graph is not acyclic; cannot compute CPM.")

    # -----------------------------
    # Forward pass: ES / EF
    # -----------------------------
    es = {n: 0.0 for n in nodes}
    ef = {n: 0.0 for n in nodes}

    # For FF/SF, we may have finish-based constraints
    max_ef_constraint = {n: 0.0 for n in nodes}

    for n in topo:
        d = durations[n]
        # respect any FF/SF-based constraints
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
                # Unknown type → treat as FS
                cand_es = ef[n] + lag
                es[succ] = max(es.get(succ, 0.0), cand_es)

    # After forward pass, re-finalize EF for all nodes
    for n in nodes:
        d = durations[n]
        ef[n] = max(es[n] + d, max_ef_constraint[n])

    project_finish = max(ef.values()) if ef else 0.0

    # -----------------------------
    # Backward pass: LS / LF
    # -----------------------------
    lf = {n: project_finish for n in nodes}
    ls = {n: project_finish - durations[n] for n in nodes}

    # Reverse topological order
    for n in reversed(topo):
        d = durations[n]
        # For each successor constraint, tighten LS/LF of n
        for succ, dep_type, lag in edges_from.get(n, []):
            if dep_type == "FS":
                # S_succ >= F_n + lag  →  F_n <= S_succ - lag
                lf[n] = min(lf[n], ls[succ] - lag)

            elif dep_type == "SS":
                # S_succ >= S_n + lag  →  S_n <= S_succ - lag
                ls[n] = min(ls[n], ls[succ] - lag)

            elif dep_type == "FF":
                # F_succ >= F_n + lag  →  F_n <= F_succ - lag
                lf[n] = min(lf[n], lf[succ] - lag)

            elif dep_type == "SF":
                # F_succ >= S_n + lag  →  S_n <= F_succ - lag
                ls[n] = min(ls[n], lf[succ] - lag)

            else:
                # treat unknown as FS
                lf[n] = min(lf[n], ls[succ] - lag)

        # recompute LS from LF and duration
        ls[n] = min(ls[n], lf[n] - d)

    # -----------------------------
    # Total float
    # -----------------------------
    total_float = {n: ls[n] - es[n] for n in nodes}

    return es, ef, ls, lf, total_float

# ---------------------------------------------------------
# COMPILE RESULTS
# ---------------------------------------------------------

def compile_results(df, bl_data, lv_data):
    ES_bl, EF_bl, LS_bl, LF_bl, Float_bl, CP_bl = bl_data
    ES_lv, EF_lv, LS_lv, LF_lv, Float_lv, CP_lv = lv_data

    df2 = df.copy()

    # Attach fields
    df2["BL_ES"] = df2["TaskID"].map(ES_bl)
    df2["BL_EF"] = df2["TaskID"].map(EF_bl)
    df2["BL_LS"] = df2["TaskID"].map(LS_bl)
    df2["BL_LF"] = df2["TaskID"].map(LF_bl)
    df2["BL_Float"] = df2["TaskID"].map(Float_bl)
    df2["BL_Critical"] = df2["TaskID"].isin(CP_bl)

    df2["LV_ES"] = df2["TaskID"].map(ES_lv)
    df2["LV_EF"] = df2["TaskID"].map(EF_lv)
    df2["LV_LS"] = df2["TaskID"].map(LS_lv)
    df2["LV_LF"] = df2["TaskID"].map(LF_lv)
    df2["LV_Float"] = df2["TaskID"].map(Float_lv)
    df2["LV_Critical"] = df2["TaskID"].isin(CP_lv)

    # Expected % complete (based on baseline)
    today = pd.Timestamp.today().normalize()
    df2["ExpectedPercent"] = (
        (today - df2["Baseline Start"]).dt.days.clip(lower=0)
        / df2["Dur_BL"].replace(0, np.nan)
    ) * 100

    df2["ExpectedPercent"] = df2["ExpectedPercent"].clip(0, 100).fillna(0)

    # Behind schedule
    df2["BehindSchedule"] = df2["PercentComplete"] < df2["ExpectedPercent"]

    # Leaf detection
    df2 = build_hierarchy(df2)

    return df2, CP_bl, CP_lv


# ---------------------------------------------------------
# EXPORTED ENTRY POINT
# ---------------------------------------------------------

def compute_dual_cpm_from_df(df_input):
    """
    Main entry after CSV or PSPLIB ingestion.
    Processes DF, runs baseline + live CPM, returns computed DF.
    """

    df = process_uploaded_dataframe(df_input)

    # 🔥 ADD THIS LINE — debug duration mismatch
    print("\n=== DEBUG: Columns entering CPM ===")
    print(df.columns.tolist())
    print(df.head(20).to_string())
    print("=== END DEBUG ===\n")

    # Live remaining duration (default)
    df["Remaining_LV"] = df["Dur_BL"] * (1 - df["PercentComplete"] / 100)

    # Baseline graph
    nodes_bl, durations_bl, edges_from_bl, edges_to_bl = build_graph(df, mode="baseline")
    es_bl, ef_bl, ls_bl, lf_bl, tf_bl = compute_cpm(nodes_bl, durations_bl, edges_from_bl, edges_to_bl)

    # Live graph
    nodes_lv, durations_lv, edges_from_lv, edges_to_lv = build_graph(df, mode="live")
    es_lv, ef_lv, ls_lv, lf_lv, tf_lv = compute_cpm(nodes_lv, durations_lv, edges_from_lv, edges_to_lv)

    # Pack tuples for compiler
    bl_data = (es_bl, ef_bl, ls_bl, lf_bl, tf_bl, [n for n in nodes_bl if tf_bl[n] == 0])
    lv_data = (es_lv, ef_lv, ls_lv, lf_lv, tf_lv, [n for n in nodes_lv if tf_lv[n] == 0])

    return compile_results(df, bl_data, lv_data)