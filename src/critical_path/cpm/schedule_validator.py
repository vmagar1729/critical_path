import pandas as pd
import re

# ------------------------------------------------------------------
# 🧱 Helper: make a consistent issue dictionary
# ------------------------------------------------------------------
def make_issue(task_id, name, severity, issue_type, description, suggestion):
    return {
        "TaskID": task_id,
        "Name": name,
        "Severity": severity,
        "IssueType": issue_type,
        "Description": description,
        "SuggestedFix": suggestion,
    }


# ------------------------------------------------------------------
# 🔍 VALID PREDECESSOR REGEX (FINAL VERSION)
# Supports:
#  - 5
#  - 12FS
#  - 12FS+2
#  - 12FS+2d
#  - 5SS-3
#  - 10, 12FS+2d, 99SS
# ------------------------------------------------------------------
PRED_PATTERN = re.compile(
    r"""
    ^\s*
    \d+                                # first task ID
    (?:FS|SS|FF|SF)?                    # optional link type
    (?:[+-]\d+\s*d?)?                  # optional lag
    (?:                                # additional predecessors
        \s*,\s*
        \d+
        (?:FS|SS|FF|SF)?
        (?:[+-]\d+\s*d?)?
    )*
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def valid_pred_format(cell):
    if cell is None or pd.isna(cell):
        return True
    s = str(cell).strip()
    if s == "":
        return True
    return bool(PRED_PATTERN.match(s))


# ------------------------------------------------------------------
# 🧠 MAIN VALIDATION ENGINE
# ------------------------------------------------------------------
def validate_schedule(df):
    issues = []

    # --- Column Normalization for MSP CSV ---
    rename_map = {
        "% Complete": "PercentComplete",
        "Outline Level": "OutlineLevel",
        "Baseline Start": "BaselineStart",
        "Baseline Finish": "BaselineFinish",
    }
    df = df.rename(columns=rename_map)

    # --- Required Columns ---
    required = [
        "TaskID", "Name", "Duration", "Start", "Finish",
        "BaselineStart", "BaselineFinish", "PercentComplete",
        "Predecessors", "WBS", "OutlineLevel"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        issues.append(
            make_issue(
                "N/A", "N/A", "critical", "MissingColumns",
                f"Missing required columns: {missing}",
                "In Microsoft Project → Insert Column → Add missing fields before exporting."
            )
        )
        return issues

    # ------------------------------------------------------------------
    # 1. TaskID validation
    # ------------------------------------------------------------------
    if df["TaskID"].isna().any():
        issues.append(
            make_issue(
                None, None, "critical", "TaskIDBlank",
                "Some TaskID values are blank.",
                "Ensure TaskID column exists and every task has a numeric ID."
            )
        )

    if not pd.to_numeric(df["TaskID"], errors="coerce").notna().all():
        issues.append(
            make_issue(
                None, None, "critical", "TaskIDNonNumeric",
                "Non-numeric TaskID values found.",
                "TaskID must be an integer. Remove text values."
            )
        )

    dups = df[df["TaskID"].duplicated()]["TaskID"].tolist()
    if dups:
        issues.append(
            make_issue(
                ", ".join(map(str, dups)), "",
                "critical", "DuplicateTaskID",
                f"Duplicate TaskIDs detected: {dups}",
                "Fix duplicate TaskIDs in MS Project: renumber tasks."
            )
        )

    # ------------------------------------------------------------------
    # 2. Duration validation
    # ------------------------------------------------------------------
    if not pd.to_numeric(df["Duration"], errors="coerce").notna().all():
        issues.append(make_issue(
            None, None, "error", "DurationInvalid",
            "Some durations contain text or invalid values.",
            "Remove values like 'TBD'. Only use numbers."
        ))

    bad_dur = df[df["Duration"] < 0]
    for _, row in bad_dur.iterrows():
        issues.append(make_issue(
            row["TaskID"], row["Name"],
            "critical", "NegativeDuration",
            f"Duration is negative ({row['Duration']}).",
            "Duration must be positive."
        ))

    # ------------------------------------------------------------------
    # 3. Date parsing
    # ------------------------------------------------------------------
    for col in ["Start", "Finish", "BaselineStart", "BaselineFinish"]:
        raw = df[col]
        parsed = pd.to_datetime(raw, errors="coerce")
        bad_count = parsed.isna().sum()
        df[col] = parsed

        if bad_count > 0:
            issues.append(make_issue(
                None, None,
                "error" if bad_count < len(df)/10 else "critical",
                "InvalidDate",
                f"{col} has {bad_count} unparseable date(s).",
                f"Fix invalid {col} values in MS Project before exporting."
            ))

    # ------------------------------------------------------------------
    # 4. Logical date rules
    # ------------------------------------------------------------------
    # Start > Finish
    bad = df[df["Start"] > df["Finish"]]
    for _, row in bad.iterrows():
        issues.append(make_issue(
            row["TaskID"], row["Name"],
            "critical", "InvalidDateOrder",
            "Start date is after Finish date.",
            "Fix Start/Finish ordering in MSP (Gantt view)."
        ))

    # BaselineStart > BaselineFinish
    bad = df[df["BaselineStart"] > df["BaselineFinish"]]
    for _, row in bad.iterrows():
        issues.append(make_issue(
            row["TaskID"], row["Name"],
            "critical", "InvalidBaselineOrder",
            "Baseline Finish is before Baseline Start.",
            "Rebaseline or correct baseline dates."
        ))

    # ------------------------------------------------------------------
    # 5. Predecessor validation
    # ------------------------------------------------------------------
    all_ids = set(df["TaskID"].astype(int))

    for _, row in df.iterrows():
        tid = row["TaskID"]
        name = row["Name"]
        cell = str(row["Predecessors"]).strip()

        if cell in ["", "nan", "None"]:
            continue

        if not valid_pred_format(cell):
            issues.append(make_issue(
                tid, name,
                "critical", "InvalidPredecessorFormat",
                f"Invalid predecessor format: '{cell}'",
                "Valid examples: 5, 12FS, 12FS+2d, 5SS-3, 10, 12FS+1d"
            ))
            continue

        # Validate referenced IDs exist
        for part in re.split(r"[;,]", cell):
            if not part.strip():
                continue

            m = re.match(r"(\d+)", part.strip())
            if m:
                pred_id = int(m.group(1))
                if pred_id not in all_ids:
                    issues.append(make_issue(
                        tid, name,
                        "error", "MissingPredecessorTask",
                        f"Task depends on missing TaskID {pred_id}.",
                        "Fix dependency: remove or correct missing TaskID."
                    ))

    return issues