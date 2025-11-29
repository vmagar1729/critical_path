import re

def load_j30_optima(path):
    """
    Load optimal makespans from PSPLIB j30opt.sm file.
    Returns dict: {instance_number : optimal_makespan}
    """
    optima = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            parts = re.findall(r"\d+", line)
            if len(parts) == 2:
                inst = int(parts[0])
                opt = int(parts[1])
                optima[inst] = opt
    return optima


def parse_instance_number(sm_filename):
    """
    Decode j301_1.sm → 1
           j301_30.sm → 30
           j302_1.sm → 31
           ...
           j3016_30.sm → 480
    """
    base = sm_filename.replace(".sm", "")
    m = re.match(r"j30(\d+)_(\d+)", base)
    if not m:
        raise ValueError(f"Invalid PSPLIB name: {sm_filename}")

    batch = int(m.group(1))      # 1–16
    index = int(m.group(2))      # 1–30

    return (batch - 1) * 30 + index