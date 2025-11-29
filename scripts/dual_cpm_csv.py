import pandas as pd
import numpy as np
from collections import defaultdict, deque
import datetime


# ---------------------------------------------------------
# FIELD CLEANUP & PREPARATION
# ---------------------------------------------------------

def process_uploaded_dataframe(df_input):
    """
    Clean and normalize a DataFrame exported from MS Project or generated from PSPLIB loader.
    Produces a uniform structure required for CPM computation.
    """

    df = df_input.copy()
    df.columns = [c.strip() for c in df.columns]

    required = [
        "TaskID", "Name", "Start", "Finish",
        "Baseline Start", "Baseline Finish",
        "Predecessors", "WBS", "Outline Level",
        "% Complete"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Fix percent complete: strip %, convert to float
    df["PercentComplete"] = (
        df["% Complete"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .replace("", "0")
        .astype(float)
    )

    # Parse date fields
    date_fields = ["Start", "Finish", "Baseline Start", "Baseline Finish"]
    for f in date_fields:
        df[f] = pd.to_datetime(df[f], errors="coerce")

    # Normalize predecessor field
    df["Predecessors"] = df["Predecessors"].fillna("").astype(str)

    # Duration fields
    df["Dur_BL"] = (df["Baseline Finish"] - df["Baseline Start"]).dt.days.clip(lower=0)
    df["Dur_LV"] = (df["Finish"] - df["Start"]).dt.days.clip(lower=0)

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


# ---------------------------------------------------------
# GRAPH CONSTRUCTION (Baseline or Live)
# ---------------------------------------------------------

def build_graph(df, mode="baseline"):
    """
    Constructs a DAG for CPM based on Baseline or Live durations.
    mode = "baseline" or "live"
    """

    # Duration selector
    if mode == "baseline":
        dur_field = "Dur_BL"
    else:
        dur_field = "Remaining_LV"

    G = defaultdict(list)
    durations = {}

    for _, row in df.iterrows():
        tid = int(row["TaskID"])
        durations[tid] = float(row.get(dur_field, 0))

        preds_raw = row["Predecessors"]
        if preds_raw.strip() == "":
            continue

        preds = [int(p) for p in preds_raw.replace(";", ",").split(",") if p.strip().isdigit()]
        for p in preds:
            G[p].append(tid)

    return {"edges": G, "durations": durations}


# ---------------------------------------------------------
# CPM ALGORITHM
# ---------------------------------------------------------

def compute_cpm(G):
    """
    G = {edges: {u:[v...]}, durations:{task:dur}}
    Returns ES, EF, LS, LF, Float, CP list.
    """

    edges = G["edges"]
    dur = G["durations"]

    # Build reverse edges
    rev = defaultdict(list)
    for u, vs in edges.items():
        for v in vs:
            rev[v].append(u)

    tasks = list(dur.keys())

    # Compute indegree for topological sort
    indeg = {t: 0 for t in tasks}
    for vs in edges.values():
        for v in vs:
            indeg[v] += 1

    # Topological order
    Q = deque([t for t in tasks if indeg[t] == 0])
    topo = []

    while Q:
        u = Q.popleft()
        topo.append(u)
        for v in edges.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                Q.append(v)

    # Detect cycle
    if len(topo) != len(tasks):
        raise ValueError("Graph contains a cycle. CPM cannot proceed.")

    # Forward pass (ES/EF)
    ES = {t: 0 for t in tasks}
    EF = {t: 0 for t in tasks}

    for t in topo:
        EF[t] = ES[t] + dur[t]
        for nxt in edges.get(t, []):
            ES[nxt] = max(ES[nxt], EF[t])

    # Backward pass (LS/LF)
    proj_end = max(EF.values()) if EF else 0
    LF = {t: proj_end for t in tasks}
    LS = {t: proj_end - dur[t] for t in tasks}

    for t in reversed(topo):
        for nxt in edges.get(t, []):
            LF[t] = min(LF[t], LS[nxt])
        LS[t] = LF[t] - dur[t]

    Float = {t: LS[t] - ES[t] for t in tasks}

    # Critical path: tasks with zero float along the longest chain
    CP = [t for t in topo if Float[t] == 0]

    return ES, EF, LS, LF, Float, CP


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

    # Live remaining duration (default)
    df["Remaining_LV"] = df["Dur_BL"] * (1 - df["PercentComplete"] / 100)

    # Baseline graph
    G_bl = build_graph(df, mode="baseline")
    bl = compute_cpm(G_bl)

    # Live graph
    G_lv = build_graph(df, mode="live")
    lv = compute_cpm(G_lv)

    # Compile
    return compile_results(df, bl, lv)