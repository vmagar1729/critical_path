import pandas as pd
import numpy as np
from typing import Dict, Any


# --------------------------------------------------------
# Utility: extract gatekeepers (tasks controlling finish)
# --------------------------------------------------------
def _get_gatekeepers(df: pd.DataFrame, float_threshold: float) -> pd.DataFrame:
    """
    Tasks that control the finish date:
      - Leaf tasks (if available)
      - Live float <= threshold
      - Remaining_LV > 0
    """
    gate = df.copy()

    # If IsLeaf exists, filter leaf tasks
    if "IsLeaf" in gate.columns:
        gate = gate[gate["IsLeaf"]]

    gate = gate[gate["Float_LV"] <= float_threshold]
    gate = gate[gate["Remaining_LV"] > 0]

    return gate


# --------------------------------------------------------
# Main recovery engine
# --------------------------------------------------------
def compute_recovery_plan(
    df: pd.DataFrame,
    float_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Given a CPM-enriched dataframe, compute how to recover schedule slip.

    Returns:
      slip: float
      recoverable: bool
      scenario_type: "none" | "single" | "multi" | "impossible"
      gatekeepers: DataFrame
      single_task: Series or None
      selected_tasks: DataFrame or None
    """

    # -----------------------------
    # Compute slip
    # -----------------------------
    bl_finish = df["BL_EF"].max()
    lv_finish = df["LV_EF"].max()

    if pd.isna(bl_finish) or pd.isna(lv_finish):
        return {
            "slip": 0.0,
            "recoverable": False,
            "scenario_type": "none",
            "gatekeepers": df.head(0),
            "single_task": None,
            "selected_tasks": None,
        }

    slip = float(lv_finish - bl_finish)

    # If not behind → nothing to fix
    if slip <= 0:
        return {
            "slip": slip,
            "recoverable": False,
            "scenario_type": "none",
            "gatekeepers": df.head(0),
            "single_task": None,
            "selected_tasks": None,
        }

    # -----------------------------
    # Find gatekeepers
    # -----------------------------
    gate = _get_gatekeepers(df, float_threshold)

    if gate.empty:
        return {
            "slip": slip,
            "recoverable": False,
            "scenario_type": "impossible",
            "gatekeepers": df.head(0),
            "single_task": None,
            "selected_tasks": None,
        }

    gate = gate.copy()

    # Ensure CriticalityWeight exists
    if "CriticalityWeight" not in gate.columns:
        gate["CriticalityWeight"] = np.where(gate["Float_LV"] == 0, 1.0, 0.5)

    # Impact factor = Remaining * weight
    gate["ImpactFactor"] = gate["Remaining_LV"] * gate["CriticalityWeight"]

    gate = gate.sort_values("ImpactFactor", ascending=False)

    total_possible = gate["Remaining_LV"].sum()

    # Not enough to recover entire slip
    if total_possible < slip - 1e-6:
        return {
            "slip": slip,
            "recoverable": False,
            "scenario_type": "impossible",
            "gatekeepers": gate,
            "single_task": None,
            "selected_tasks": None,
        }

    # -----------------------------
    # Single-task recovery scenario
    # -----------------------------
    top = gate.iloc[0]

    if top["Remaining_LV"] >= slip:
        row = top.copy()
        row["RequiredCut"] = slip
        row["NewRemaining"] = top["Remaining_LV"] - slip

        single_df = pd.DataFrame([row])

        return {
            "slip": slip,
            "recoverable": True,
            "scenario_type": "single",
            "gatekeepers": gate,
            "single_task": single_df.iloc[0],
            "selected_tasks": single_df,
        }

    # -----------------------------
    # Multi-task greedy recovery
    # -----------------------------
    need = slip
    selected = []

    for _, r in gate.iterrows():
        if need <= 0:
            break

        max_cut = r["Remaining_LV"]
        cut = min(max_cut, need)
        need -= cut

        rr = r.copy()
        rr["RequiredCut"] = cut
        rr["NewRemaining"] = rr["Remaining_LV"] - cut
        selected.append(rr)

    selected_df = pd.DataFrame(selected)

    return {
        "slip": slip,
        "recoverable": True,
        "scenario_type": "multi",
        "gatekeepers": gate,
        "single_task": None,
        "selected_tasks": selected_df,
    }