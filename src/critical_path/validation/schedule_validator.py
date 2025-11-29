import pandas as pd

from critical_path.cpm.dual_cpm_csv import compute_dual_cpm_from_df


# -------------------------------------------------------------------
# Helper: create a standardized issue entry
# -------------------------------------------------------------------
def _issue(task_id, name, category, severity, issue_type, description, suggestion):
    return {
        "TaskID": task_id,
        "Name": name,
        "Category": category,
        "Severity": severity,
        "IssueType": issue_type,
        "Description": description,
        "SuggestedFix": suggestion,
    }


# -------------------------------------------------------------------
# MAIN VALIDATOR
# -------------------------------------------------------------------
def validate_schedule(df, status_date=None):
    """
    Validates schedule data and returns a dataframe of issues with:
    TaskID, Name, Category, Severity, IssueType, Description, SuggestedFix.
    """

    issues = []

    # Normalize predecessors
    def parse_preds(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [int(p) for p in str(x).split(",") if p.strip().isdigit()]

    df["PredList"] = df["Predecessors"].apply(parse_preds)

    task_ids = set(df["TaskID"])

    # Status Date
    if status_date is None:
        status_date = pd.Timestamp.now()

    # Build successor list
    successors = {t: [] for t in task_ids}
    for _, row in df.iterrows():
        for p in row["PredList"]:
            if p in successors:
                successors[p].append(row["TaskID"])


    # ============================================================
    # RULE 1 — Missing Predecessors
    # ============================================================
    for _, row in df.iterrows():
        if len(row["PredList"]) == 0 and row["TaskID"] != df["TaskID"].min():
            issues.append(_issue(
                row["TaskID"], row["Name"], "Dependencies", "Warning",
                "MissingPredecessors",
                "Task has no predecessors; may start prematurely.",
                "Verify sequencing and add logical predecessor links."
            ))


    # ============================================================
    # RULE 2 — Invalid Predecessor References
    # ============================================================
    for _, row in df.iterrows():
        for p in row["PredList"]:
            if p not in task_ids:
                issues.append(_issue(
                    row["TaskID"], row["Name"], "Dependencies", "Critical",
                    "InvalidPredecessor",
                    f"Predecessor {p} is not a valid TaskID.",
                    "Correct the predecessor reference."
                ))


    # ============================================================
    # RULE 3 — Circular Dependencies
    # ============================================================
    graph = {int(t): [] for t in task_ids}
    for _, row in df.iterrows():
        for p in row["PredList"]:
            graph[p].append(row["TaskID"])

    visited = set()
    stack = set()

    def dfs(node):
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for nxt in graph[node]:
            if dfs(nxt):
                return True
        stack.remove(node)
        return False

    for node in task_ids:
        visited.clear()
        stack.clear()
        if dfs(node):
            issues.append(_issue(
                node, df.loc[df["TaskID"] == node, "Name"].iloc[0],
                "Dependencies", "Critical", "CycleDetected",
                "Circular dependency detected.",
                "Remove the cycle by correcting task relationships."
            ))


    # ============================================================
    # RULE 4 — Orphan Tasks
    # ============================================================
    for _, row in df.iterrows():
        if len(row["PredList"]) == 0 and len(successors[row["TaskID"]]) == 0:
            issues.append(_issue(
                row["TaskID"], row["Name"], "Dependencies", "Warning",
                "OrphanTask",
                "Task has no predecessors AND no successors.",
                "If this task matters, add sequencing. Otherwise convert to a note/milestone."
            ))


    # ============================================================
    # RULE 5 — % Complete Consistency
    # ============================================================
    for _, row in df.iterrows():
        pc = row["PercentComplete"]
        start = row["Start"]
        finish = row["Finish"]

        if pc > 0 and pd.isna(start):
            issues.append(_issue(
                row["TaskID"], row["Name"], "Dates", "Warning",
                "MissingStartDate",
                "Task has progress but Start date is missing.",
                "Set Start date based on when work actually began."
            ))

        if pc >= 100 and pd.isna(finish):
            issues.append(_issue(
                row["TaskID"], row["Name"], "Dates", "Critical",
                "MissingFinishDate",
                "Task marked complete but Finish date is missing.",
                "Set Finish date to the actual completion date."
            ))


    # ============================================================
    # RULE 6 — Duration Validation
    # ============================================================
    for _, row in df.iterrows():
        if row["Dur_BL"] < 0:
            issues.append(_issue(
                row["TaskID"], row["Name"], "Duration", "Critical",
                "NegativeDuration",
                f"Duration is negative: {row['Dur_BL']}.",
                "Correct the duration to a positive value."
            ))

        if row["Dur_BL"] == 0 and len(successors[row["TaskID"]]) > 0:
            issues.append(_issue(
                row["TaskID"], row["Name"], "Duration", "Warning",
                "ZeroDurationNonMilestone",
                "Task has 0 duration but has successors.",
                "If this is a milestone, convert it. Otherwise add a realistic duration."
            ))


    # ============================================================
    # RULE 7 — Baseline Date Consistency
    # ============================================================
    for _, row in df.iterrows():
        if row["Baseline Finish"] < row["Baseline Start"]:
            issues.append(_issue(
                row["TaskID"], row["Name"], "Dates", "Critical",
                "BaselineDateError",
                "Baseline Finish precedes Baseline Start.",
                "Correct the baseline dates."
            ))


    # ============================================================
    # DONE
    # ============================================================
    return pd.DataFrame(issues)


# -------------------------------------------------------------------
# CSV Export Wrapper
# -------------------------------------------------------------------
def validate_and_export(df, out_csv="schedule_validation_results.csv", status_date=None):
    issues_df = validate_schedule(df, status_date=status_date)
    issues_df.to_csv(out_csv, index=False)
    return issues_df


df = pd.read_csv(
    "/Users/vivekmagar/PycharmProjects/critical_path_git/synthetic_schedules/schedule1.csv",
    parse_dates=["Start", "Finish", "Baseline Start", "Baseline Finish"]
)
validate_and_export(compute_dual_cpm_from_df(df))