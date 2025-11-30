import pytest
import pandas as pd
import numpy as np
from critical_path import process_uploaded_dataframe, build_graph, compute_cpm, parse_predecessor_cell


# ----------------------------------------------------------------
# 1. PARSING TESTS
# ----------------------------------------------------------------
def test_parse_predecessor_cell():
    # Test Standard FS
    assert parse_predecessor_cell("10") == [(10, "FS", 0.0)]
    assert parse_predecessor_cell("10FS") == [(10, "FS", 0.0)]

    # Test Lags (Positive/Negative)
    assert parse_predecessor_cell("10FS+2d") == [(10, "FS", 2.0)]
    assert parse_predecessor_cell("10FS - 3 d") == [(10, "FS", -3.0)]

    # Test Types (SS, FF, SF)
    assert parse_predecessor_cell("20SS+5") == [(20, "SS", 5.0)]
    assert parse_predecessor_cell("30FF") == [(30, "FF", 0.0)]

    # Test Multiple Dependencies
    res = parse_predecessor_cell("10FS, 20SS+2")
    assert (10, "FS", 0.0) in res
    assert (20, "SS", 2.0) in res


# ----------------------------------------------------------------
# 2. CORE CPM LOGIC TESTS
# ----------------------------------------------------------------

def run_cpm_on_data(data):
    """Helper to run the full pipeline on a dict/list structure"""
    df = pd.DataFrame(data)
    # Ensure standard columns exist
    if "Predecessors" not in df.columns:
        df["Predecessors"] = ""
    if "PercentComplete" not in df.columns:
        df["PercentComplete"] = 0
    if "Owner" not in df.columns:
        df["Owner"] = "Test"

    # normalize
    df = process_uploaded_dataframe(df)

    # build and compute
    nodes, durations, edges_from, edges_to = build_graph(df, mode="baseline")
    es, ef, ls, lf, tf = compute_cpm(nodes, durations, edges_from, edges_to)
    return es, ef, ls, lf, tf


def test_simple_fs_chain():
    """
    Task 1 (Dur 5) -> Task 2 (Dur 3)
    Expected:
      T1: ES=0, EF=5
      T2: ES=5, EF=8
    """
    data = [
        {"TaskID": 1, "Name": "A", "Duration": 5, "Predecessors": ""},
        {"TaskID": 2, "Name": "B", "Duration": 3, "Predecessors": "1"}
    ]
    es, ef, ls, lf, tf = run_cpm_on_data(data)

    assert es[1] == 0.0 and ef[1] == 5.0
    assert es[2] == 5.0 and ef[2] == 8.0
    assert tf[1] == 0.0 and tf[2] == 0.0  # Both critical


def test_ss_dependency_with_lag():
    """
    Task 1 (Dur 10)
    Task 2 (Dur 5) depends on 1SS+2
    Expected:
      T2 ES = T1 ES + 2 = 0 + 2 = 2.0
    """
    data = [
        {"TaskID": 1, "Name": "A", "Duration": 10},
        {"TaskID": 2, "Name": "B", "Duration": 5, "Predecessors": "1SS+2"}
    ]
    es, ef, _, _, _ = run_cpm_on_data(data)

    assert es[2] == 2.0
    # T2 finishes at 2+5=7. T1 finishes at 10. Project ends at 10.
    # T2 Float? T2 can finish as late as 10 (LF).
    # T2 LS = LF - Dur = 10 - 5 = 5.
    # T2 Float = LS - ES = 5 - 2 = 3.


def test_ff_dependency():
    """
    Task 1 (Dur 10)
    Task 2 (Dur 2) depends on 1FF
    (Task 2 cannot finish until Task 1 finishes)
    Expected:
      T1 EF = 10
      T2 EF >= T1 EF (10).
      Since T2 Dur is 2, T2 ES must be at least 8 to finish at 10.
    """
    data = [
        {"TaskID": 1, "Name": "A", "Duration": 10},
        {"TaskID": 2, "Name": "B", "Duration": 2, "Predecessors": "1FF"}
    ]
    es, ef, _, _, _ = run_cpm_on_data(data)

    assert ef[1] == 10.0
    assert ef[2] == 10.0  # Driven by 1FF constraint
    assert es[2] == 8.0  # Derived from EF - Dur


def test_circular_dependency_error():
    """
    A -> B -> A loop should raise ValueError
    """
    data = [
        {"TaskID": 1, "Name": "A", "Duration": 5, "Predecessors": "2"},
        {"TaskID": 2, "Name": "B", "Duration": 5, "Predecessors": "1"}
    ]
    with pytest.raises(ValueError, match="Graph is not acyclic"):
        run_cpm_on_data(data)


def test_multiple_paths_convergence():
    """
    A (5) -> C (2)
    B (10) -> C (2)
    C should start at max(5, 10) = 10
    """
    data = [
        {"TaskID": 1, "Name": "A", "Duration": 5},
        {"TaskID": 2, "Name": "B", "Duration": 10},
        {"TaskID": 3, "Name": "C", "Duration": 2, "Predecessors": "1, 2"}
    ]
    es, ef, ls, lf, tf = run_cpm_on_data(data)

    assert es[3] == 10.0
    assert tf[1] == 5.0  # A has 5 days float (can finish at 10)
    assert tf[2] == 0.0  # B is critical