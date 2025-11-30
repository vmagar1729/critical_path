# critical_path/cpm/prediction_engine.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


# -----------------------------
# Small helpers
# -----------------------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize(series: pd.Series) -> pd.Series:
    s = _safe_numeric(series).fillna(0.0)
    span = s.max() - s.min()
    if span <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / span


# -----------------------------
# 0. Ensure prediction fields
# -----------------------------

PREDICTION_REQUIRED = [
    "TaskID",
    "Name",
    "Owner",
    "Remaining_LV",
    "Float_LV",
    "SlippageExposure",
    "ScheduleVariance",
    "PercentComplete",
]


def ensure_prediction_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive: makes sure the frame has what prediction functions expect.
    If things are missing, fills them with safe defaults so the UI
    degrades gracefully instead of exploding.
    """
    df2 = df.copy()

    # Owner normalization
    if "Owner" not in df2.columns:
        df2["Owner"] = "Unassigned"
    df2["Owner"] = df2["Owner"].fillna("Unassigned").astype(str)

    # Core numeric fields with defaults
    for col, default in [
        ("Remaining_LV", 0.0),
        ("Float_LV", 0.0),
        ("SlippageExposure", 0.0),
        ("ScheduleVariance", 0.0),
        ("PercentComplete", 0.0),
    ]:
        if col not in df2.columns:
            df2[col] = default
        df2[col] = _safe_numeric(df2[col]).fillna(default)

    # Optional creep info
    if "HasDurationCreep" not in df2.columns:
        df2["HasDurationCreep"] = False
    if "DurationCreep" not in df2.columns:
        df2["DurationCreep"] = 0.0
    df2["DurationCreep"] = _safe_numeric(df2["DurationCreep"]).fillna(0.0)

    # Critical flags
    if "IsCritical_LV" not in df2.columns:
        df2["IsCritical_LV"] = df2["Float_LV"] == 0

    return df2


# -----------------------------
# 1. Per-task sigma estimation
# -----------------------------

def estimate_task_sigmas(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic "ML-flavored" sigma per task.

    Uses:
      - Owner volatility (spread of schedule variance for that owner)
      - Whether the task is already behind
      - Slack position (near-critical vs safe)
      - Historical duration creep
    """
    df2 = ensure_prediction_ready(df)

    # Owner volatility: std dev of abs(schedule variance)
    owner_stats = (
        df2.groupby("Owner")["ScheduleVariance"]
        .apply(lambda s: np.nanstd(np.abs(_safe_numeric(s))))
        .replace({np.nan: 0.0})
    )
    owner_norm = _normalize(owner_stats)

    df2["OwnerVolatility"] = df2["Owner"].map(owner_norm).fillna(0.0)

    # Flags
    df2["IsBehind"] = (df2["ScheduleVariance"] < 0).astype(float)
    df2["NearCritical"] = ((df2["Float_LV"] >= 0) & (df2["Float_LV"] <= 1.0)).astype(float)
    df2["HasCreepNum"] = (df2.get("HasDurationCreep", False)).astype(float)

    creep_norm = _normalize(df2.get("DurationCreep", 0.0))

    # Risk score in [0, ~1]
    risk = (
        0.35 * df2["OwnerVolatility"]
        + 0.25 * df2["IsBehind"]
        + 0.25 * df2["NearCritical"]
        + 0.15 * creep_norm
    )
    risk = risk.clip(0.0, 1.0)

    # Map risk to sigma range: 5%–30% relative std dev
    sigma = 0.05 + 0.25 * risk
    return sigma


# -----------------------------
# 2. Monte-Carlo simulation
# -----------------------------

def run_monte_carlo(
    df: pd.DataFrame,
    n_iter: int = 500,
    critical_float_threshold: float = 0.5,
    random_state: int | None = None,
) -> Dict[str, Any]:
    """
    Monte-Carlo on the *remaining* work of tasks that actually control the date.

    We don't recompute full CPM every run (too slow for Streamlit),
    we approximate project finish slip as:

        slip ≈ sum(perturbed remaining on critical / near-critical tasks)
              − sum(current remaining on those same tasks)

    and apply that slip to the current live finish.

    Returns:
      {
        "baseline_finish": float,
        "live_finish": float,
        "samples": np.ndarray (n_iter,),
        "slip_samples": np.ndarray (n_iter,),
        "p50": float,
        "p80": float,
        "p90": float,
        "prob_on_or_before_baseline": float,
        "used_tasks": pd.DataFrame
      }
    """
    rng = np.random.default_rng(random_state)
    df2 = ensure_prediction_ready(df)

    if "LV_EF" not in df2.columns or "BL_EF" not in df2.columns:
        raise ValueError("Dataframe must contain BL_EF and LV_EF to run Monte-Carlo.")

    bl_finish = float(df2["BL_EF"].max())
    lv_finish = float(df2["LV_EF"].max())

    # Gate tasks: remaining work + low float
    gate = df2[
        (df2["Remaining_LV"] > 0)
        & (df2["Float_LV"] <= critical_float_threshold)
    ].copy()

    if gate.empty:
        # Nothing left that actually moves the date
        samples = np.full(n_iter, lv_finish)
        return {
            "baseline_finish": bl_finish,
            "live_finish": lv_finish,
            "samples": samples,
            "slip_samples": samples - bl_finish,
            "p50": float(lv_finish),
            "p80": float(lv_finish),
            "p90": float(lv_finish),
            "prob_on_or_before_baseline": float(lv_finish <= bl_finish),
            "used_tasks": gate,
        }

    base_remaining = gate["Remaining_LV"].to_numpy(dtype=float)
    sigmas = estimate_task_sigmas(gate).to_numpy(dtype=float)

    # Draw multiplicative factors ~ LogNormal(0, sigma)
    # median = 1.0, variance driven by sigma
    factors = rng.lognormal(mean=0.0, sigma=sigmas, size=(n_iter, len(gate)))

    sim_remaining = base_remaining * factors
    sim_total = sim_remaining.sum(axis=1)

    base_total = float(base_remaining.sum())
    extra = sim_total - base_total   # how many extra "days" appear on gatekeepers

    # Approximate project finish distribution
    finish_samples = lv_finish + extra

    p50 = float(np.percentile(finish_samples, 50))
    p80 = float(np.percentile(finish_samples, 80))
    p90 = float(np.percentile(finish_samples, 90))

    prob_on_or_before_baseline = float(np.mean(finish_samples <= bl_finish))

    return {
        "baseline_finish": bl_finish,
        "live_finish": lv_finish,
        "samples": finish_samples,
        "slip_samples": finish_samples - bl_finish,
        "p50": p50,
        "p80": p80,
        "p90": p90,
        "prob_on_or_before_baseline": prob_on_or_before_baseline,
        "used_tasks": gate,
    }


# -----------------------------
# 3. Slack burn-down projection
# -----------------------------

def compute_slack_burndown(
    df: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Approximates how float disappears as the project advances.

    We:
      - sort tasks by live early start (LV_ES) if available, else BL_ES, else TaskID
      - treat cumulative baseline duration as "progress"
      - track remaining positive float as we move forward

    Returns a dataframe with:
      ProgressPct, RemainingFloat
    """
    df2 = ensure_prediction_ready(df)

    if "Dur_BL" in df2.columns:
        dur = _safe_numeric(df2["Dur_BL"]).clip(lower=0).fillna(0.0)
    else:
        dur = _safe_numeric(df2.get("Duration", 0.0)).clip(lower=0).fillna(0.0)

    df2["DurForProgress"] = dur

    # Ordering: LV_ES > BL_ES > TaskID
    order_cols = []
    for col in ["LV_ES", "BL_ES"]:
        if col in df2.columns:
            order_cols.append(col)
    if not order_cols:
        order_cols = ["TaskID"]

    df2 = df2.sort_values(order_cols)

    df2["CumDur"] = df2["DurForProgress"].cumsum()
    total_dur = float(df2["DurForProgress"].sum())
    if total_dur <= 0:
        # Nothing useful
        return pd.DataFrame({"ProgressPct": [0.0, 100.0], "RemainingFloat": [0.0, 0.0]})

    df2["ProgressPct"] = df2["CumDur"] / total_dur * 100.0

    # Remaining positive float "ahead" of each point
    # We use Float_LV as today's view of slack
    df2["Float_LV"] = _safe_numeric(df2.get("Float_LV", 0.0)).fillna(0.0)

    grid = np.linspace(0.0, 100.0, num=bins + 1)
    rows = []

    for g in grid:
        # tasks not yet "crossed" at this progress level
        remaining = df2[df2["ProgressPct"] >= g]
        rem_float = float(remaining["Float_LV"].clip(lower=0.0).sum())
        rows.append({"ProgressPct": float(g), "RemainingFloat": rem_float})

    return pd.DataFrame(rows)


# -----------------------------
# 4. Owner overload forecast
# -----------------------------

def compute_owner_overload(
    df: pd.DataFrame,
    critical_float_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Owner-level forecast of who is going to scream first.

    Returns columns:
      Owner, TotalRemaining, CriticalRemaining, OverloadIndex
    """
    df2 = ensure_prediction_ready(df)

    df2["IsCriticalOrNear"] = (df2["Float_LV"] <= critical_float_threshold).astype(int)

    grp = (
        df2.groupby("Owner", dropna=False)
        .agg(
            TotalRemaining=("Remaining_LV", "sum"),
            CriticalRemaining=("Remaining_LV", lambda s: float(s[df2.loc[s.index, "IsCriticalOrNear"] == 1].sum())),
        )
        .reset_index()
    )

    grp["TotalRemaining"] = _safe_numeric(grp["TotalRemaining"]).clip(lower=0.0)
    grp["CriticalRemaining"] = _safe_numeric(grp["CriticalRemaining"]).clip(lower=0.0)

    eps = 1e-6
    grp["OverloadIndex"] = grp["CriticalRemaining"] / (grp["TotalRemaining"] + eps)

    # Highest overload first
    grp = grp.sort_values(["OverloadIndex", "CriticalRemaining"], ascending=[False, False])

    return grp


# -----------------------------
# 5. Risk clustering (toy ML)
# -----------------------------

def cluster_risks(df: pd.DataFrame, n_clusters: int = 3, max_iter: int = 25) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tiny k-means implementation over a simple risk feature vector:

      - Remaining_LV (normalized)
      - SlippageExposure (normalized)
      - IsBehind (0/1)
      - IsCritical (0/1)

    Returns:
      df_with_clusters, {"centroids": centroids, "cluster_risk_rank": rank_dict}
    """
    df2 = ensure_prediction_ready(df).copy()

    rem_norm = _normalize(df2["Remaining_LV"])
    slip_norm = _normalize(df2["SlippageExposure"])
    is_behind = (df2["ScheduleVariance"] < 0).astype(float)
    is_crit = (df2["Float_LV"] <= 0.0).astype(float)

    X = np.vstack(
        [
            rem_norm.to_numpy(dtype=float),
            slip_norm.to_numpy(dtype=float),
            is_behind.to_numpy(dtype=float),
            is_crit.to_numpy(dtype=float),
        ]
    ).T

    if len(df2) == 0:
        df2["RiskCluster"] = -1
        return df2, {"centroids": np.zeros((0, 4)), "cluster_risk_rank": {}}

    # Clamp n_clusters
    k = min(n_clusters, len(df2))
    rng = np.random.default_rng(42)

    # Init centroids as random samples
    init_idx = rng.choice(len(df2), size=k, replace=False)
    centroids = X[init_idx]

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Recompute
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # Re-seed dead cluster
                new_centroids[j] = X[rng.integers(0, len(X))]
            else:
                new_centroids[j] = X[mask].mean(axis=0)

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    df2["RiskClusterRaw"] = labels

    # Rank clusters by mean SlippageExposure
    cluster_risk = (
        df2.groupby("RiskClusterRaw")["SlippageExposure"]
        .mean()
        .sort_values(ascending=False)
    )

    rank_map = {cluster_id: rank for rank, cluster_id in enumerate(cluster_risk.index)}
    # 0 = highest risk, 1 = medium, etc.
    df2["RiskCluster"] = df2["RiskClusterRaw"].map(rank_map)

    info = {
        "centroids": centroids,
        "cluster_risk_rank": rank_map,
    }

    return df2, info


# -----------------------------
# 6. Executive narrative
# -----------------------------

def generate_executive_narrative(
    mc_result: Dict[str, Any],
    owner_overload_df: pd.DataFrame,
    risk_cluster_df: pd.DataFrame,
) -> str:
    """
    Cheap, offline "AI narrative":

    Takes:
      - Monte-Carlo summary
      - owner overload table
      - df with RiskCluster

    Returns a multi-paragraph string that sounds like something
    you'd say with a straight face to management.
    """
    bl = mc_result.get("baseline_finish", np.nan)
    lv = mc_result.get("live_finish", np.nan)
    p80 = mc_result.get("p80", np.nan)
    p90 = mc_result.get("p90", np.nan)
    prob_on_or_before = mc_result.get("prob_on_or_before_baseline", 0.0)

    slip_mean = float(np.mean(mc_result.get("slip_samples", [0.0])))

    # Owner hot-spots
    top_owner_line = "No specific owner concentration detected."
    if not owner_overload_df.empty:
        hot = owner_overload_df.iloc[0]
        top_owner_line = (
            f"The highest concentration of schedule-sensitive work is with **{hot['Owner']}**, "
            f"who holds about **{hot['CriticalRemaining']:.1f} days** of critical / near-critical "
            f"remaining effort (overload index ~{hot['OverloadIndex']:.2f})."
        )

    # High-risk tasks
    high_risk_tasks = risk_cluster_df[risk_cluster_df["RiskCluster"] == 0].copy()
    high_risk_tasks = high_risk_tasks.sort_values(
        "SlippageExposure", ascending=False
    ).head(5)

    if high_risk_tasks.empty:
        risk_line = "No individual tasks stand out as extreme risk drivers in the current model."
    else:
        names = ", ".join(
            f"{int(r.TaskID)} – {r.Name}" for _, r in high_risk_tasks.iterrows()
        )
        risk_line = (
            "The main schedule-risk drivers (by modeled slippage exposure) are: "
            + names
            + "."
        )

    # Narrative
    lines = []

    # Status
    lines.append(
        f"**Schedule outlook.** The deterministic plan currently finishes at about **day {lv:.1f}**, "
        f"versus a baseline target of **day {bl:.1f}**. Under modeled uncertainty, the "
        f"projected P80 finish is **day {p80:.1f}** and P90 is **day {p90:.1f}**. "
        f"The probability of landing on or before the original baseline date is about "
        f"**{prob_on_or_before * 100:.0f}%**."
    )

    # Slippage interpretation
    if slip_mean > 0.5:
        lines.append(
            f"On average, simulations add roughly **{slip_mean:.1f} days** of additional slip "
            f"on top of today’s live plan, primarily from variability in remaining work on "
            f"critical and near-critical tasks."
        )
    else:
        lines.append(
            "Modeled slippage on the remaining critical chain is modest; most scenarios hover "
            "close to today’s live finish, with limited tail risk."
        )

    # Owner overload
    lines.append(f"**Ownership & load.** {top_owner_line}")

    # Risk clusters
    lines.append(f"**Risk clusters.** {risk_line}")

    # Closing
    lines.append(
        "In plain language: if you want to protect the go-live date, focus interventions on "
        "the owners and tasks listed above before broadening the conversation. Randomly adding "
        "people to low-risk work will not move the date."
    )

    return "\n\n".join(lines)