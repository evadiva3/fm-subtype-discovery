from __future__ import annotations

import os
import sys
from pathlib import Path

RP=Path(os.environ.get("FM_REPO_ROOT") or Path(__file__).resolve().parents[2])
if str(RP) not in sys.path:
    sys.path.insert(0,str(RP))

RUNS=Path(os.environ.get("FM_RUNS_ROOT") or (RP/"results"/"handoff_20260802"))
CAN=Path(os.environ.get("FM_CANON_DIR") or (RP/"data"/"outputs"))


def out(name:str)->Path:
    p=RUNS/name
    p.mkdir(parents=True,exist_ok=True)
    return p


def detection_curve()->Path:
    for c in (RUNS/"positive_control_detection_curve.csv",
              RP/"results"/"positive_control_detection_curve.csv",
              RUNS/"audit_gap_fills"/"detection_floor_separation_units_selected.csv"):
        if c.exists():
            return c
    raise FileNotFoundError(
        "positive-control detection curve not found. Place it at "
        f"{RUNS/'positive_control_detection_curve.csv'} or regenerate the "
        "detection-floor artifact.")
