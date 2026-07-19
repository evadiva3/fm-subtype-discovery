#experimental ignore for now

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from config import config

ROOT=Path(__file__).resolve().parents[3]/"resting_state_dep_data"
CONF_PAT="{s}/{s}_rest_confounds.tsv"
TS_PAT="{s}/{s}_rest_ts.npy"
FD_COL="framewise_displacement"
SUBTYPE_AMBIG=None

def _subs():
    return sorted(f.name for f in ROOT.iterdir() if f.is_dir() and not f.name.startswith("top_") and f.name!="excluded")
def complete(sid):
    return (ROOT/TS_PAT.format(s=sid)).exists() and (ROOT/CONF_PAT.format(s=sid)).exists()
def motion_ok(sid):
    p=ROOT/CONF_PAT.format(s=sid)
    if not p.exists():
        return False
    c=pd.read_csv(p,sep="\t")
    if FD_COL not in c.columns:
        return False
    fd=c[FD_COL].fillna(0).to_numpy()
    frac=float(np.mean(fd>config.fdThreshold))
    return frac<config.fdFractionThreshold
def get_included_subjects_mdd():
    keep=[]
    for s in _subs():
        if not complete(s):
            continue
        if not motion_ok(s):
            continue
        if SUBTYPE_AMBIG is not None and s in SUBTYPE_AMBIG:
            continue
        keep.append(s)
    return sorted(keep)
