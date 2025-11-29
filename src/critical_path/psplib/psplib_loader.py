import re
import pandas as pd


def parse_sm_file(path):
    """
    Nuclear-hardened PSPLIB .sm parser.
    Extracts TaskID, Duration, Predecessors using integer-only extraction.
    Immune to whitespace corruption and unicode spacing issues.
    """

    succ_map = {}      # job -> list of successors
    durations = {}     # job -> duration

    state = None

    with open(path, "r") as f:
        for raw in f:
            # Normalize ALL whitespace, including unicode NBSP etc.
            line = re.sub(r"\s+", " ", raw.strip())

            if not line or line.startswith("*"):
                continue

            lower = line.lower()

            # Switch parser states
            if lower.startswith("precedence relations"):
                state = "prec_header"
                continue
            if lower.startswith("requests/durations"):
                state = "dur_header"
                continue

            # Skip header rows inside each section
            if state == "prec_header":
                if lower.startswith("jobnr."):
                    state = "prec"
                continue

            if state == "dur_header":
                if lower.startswith("jobnr."):
                    state = "dur"
                    continue
                if line.startswith("---"):
                    continue

            # ==========================
            # PRECEDENCE SECTION
            # ==========================
            if state == "prec":
                # Extract all integers
                ints = re.findall(r"\d+", line)
                if len(ints) < 3:
                    continue

                job = int(ints[0])
                num_succ = int(ints[2])

                succs = []
                if num_succ > 0 and len(ints) > 3:
                    succs = [int(x) for x in ints[3:]]

                succ_map[job] = succs

            # ==========================
            # DURATION SECTION
            # ==========================
            elif state == "dur":

                # Split on whitespace
                parts = line.split()

                # Reject anything shorter than jobnr mode duration
                if len(parts) < 3:
                    continue

                # First token MUST be a job number
                if not parts[0].isdigit():
                    continue

                job = int(parts[0])

                # Only accept duration rows for known jobs (from precedence section)
                if job not in succ_map:
                    continue

                # Third token MUST be the duration
                if not parts[2].isdigit():
                    continue

                dur = int(parts[2])
                durations[job] = dur

    # ================================================
    # Build predecessor map (invert successor map)
    # ================================================
    all_jobs = sorted(set(durations.keys()) | set(succ_map.keys()))
    pred_map = {j: [] for j in all_jobs}

    for j, succs in succ_map.items():
        for s in succs:
            pred_map[s].append(j)

    # ================================================
    # Construct DataFrame
    # ================================================
    rows = []
    for j in all_jobs:
        preds = pred_map.get(j, [])
        preds_str = ",".join(str(p) for p in sorted(preds)) if preds else ""
        dur = durations.get(j, 0)
        rows.append((j, dur, preds_str))

    df = pd.DataFrame(rows, columns=["TaskID", "Duration", "Predecessors"])
    df = df.sort_values("TaskID").reset_index(drop=True)
    return df


def sm_to_cpm_df(sm_path, baseline_start="2025-01-01"):
    """
    Convert a PSPLIB .sm file into a DataFrame compatible with compute_dual_cpm_from_df.
    """

    df = parse_sm_file(sm_path)

    base = pd.Timestamp(baseline_start)

    # baseline dates: same start date, finish = start + duration
    df["Baseline Start"] = base
    df["Baseline Finish"] = base + pd.to_timedelta(df["Duration"], unit="D")

    # live dates are unknown
    df["Start"] = pd.NaT
    df["Finish"] = pd.NaT

    # simple WBS + outline
    df["WBS"] = df["TaskID"].astype(str)
    df["Outline Level"] = 1

    # project % complete as Project-style strings ("0%")
    df["% Complete"] = "0%"

    # name column
    df["Name"] = df["TaskID"].apply(lambda x: f"Task {x}")

    return df

import re

def load_j30_optimal_durations(opt_path):
    """
    Parse j30opt.sm containing optimal makespans for all J30 instances.

    File format lines (after header):
        set_id   instance_id   optimal_makespan   cpu_time

    We map each (set_id, instance_id) pair to a filename like:
        j301_1.sm, j301_2.sm, ..., j3048_10.sm

    Returns:
        dict: { "j301_1.sm": 62, "j301_2.sm": 39, ..., "j3048_10.sm": 54 }
    """
    optima = {}

    with open(opt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or not line[0].isdigit():
                # skip headers / separators
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                set_id = int(parts[0])       # 1..48
                inst_id = int(parts[1])      # 1..10
                makespan = int(parts[2])     # optimal project duration
            except ValueError:
                continue

            # Match your file naming: j30{set_id}_{inst_id}.sm
            fname = f"j30{set_id}_{inst_id}.sm"
            optima[fname] = makespan

    return optima