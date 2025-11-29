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
                # Extract all integers
                ints = re.findall(r"\d+", line)
                # Format is: jobnr, mode, duration, r1, r2, ...
                if len(ints) < 3:
                    continue

                job = int(ints[0])
                dur = int(ints[2])
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

    df = df[
        [
            "TaskID",
            "Name",
            "Start",
            "Finish",
            "Baseline Start",
            "Baseline Finish",
            "Predecessors",
            "WBS",
            "Outline Level",
            "% Complete",
        ]
    ]

    return df