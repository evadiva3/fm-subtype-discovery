import sys
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

_R=Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent,_R,_R/"src",_R/"models",_R/"preprocessing"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
from config import config
from dataset_mdd import MDD_ROOT,MDD_TR,TS_PAT

FMRIPREP=MDD_ROOT.parent/"fmriprep_ds002748_out"
BOLD_PAT="{s}/func/{s}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
CONF_PAT="{s}/func/{s}_task-rest_desc-confounds_timeseries.tsv"

def parc(sid):
    bp=FMRIPREP/BOLD_PAT.format(s=sid)
    cp=FMRIPREP/CONF_PAT.format(s=sid)
    if not (bp.exists() and cp.exists()):
        return None
    c=pd.read_csv(cp,sep="\t")
    cc=[x for x in config.confoundColumns if x in c.columns]
    m=NiftiLabelsMasker(labels_img=str(config.masker),high_pass=config.highPassCutoff,t_r=MDD_TR,standardize=False)
    ts=m.fit_transform(str(bp),confounds=c[cc].fillna(0))
    op=MDD_ROOT/TS_PAT.format(s=sid)
    op.parent.mkdir(parents=True,exist_ok=True)
    np.save(op,ts)
    return ts.shape

def main():
    subs=sorted(d.name for d in FMRIPREP.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    ok=0
    for s in subs:
        r=parc(s)
        if r is None:
            print(f"skip {s} (missing fmriprep output)")
        else:
            ok+=1
            print(f"{s} -> {r}")
    print(f"done: {ok}/{len(subs)} -> {MDD_ROOT}/<sub>/<sub>_rest_ts.npy")

if __name__=="__main__":
    main()
