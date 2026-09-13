from __future__ import annotations
import os,glob,warnings
from _paths import RP
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from config import config

CONDS=[c.replace(" ","") for c in config.conditions]

def fc_from_ts(ts):
    r=ConnectivityMeasure(kind="correlation",cov_estimator=LedoitWolf()).fit_transform([ts])[0]
    z=np.arctanh(r)
    np.fill_diagonal(z,0)
    z[~np.isfinite(z)]=0
    return z

def nodes_from_ts(ts):
    df=pd.DataFrame(ts)
    mean=df.mean(axis=0).to_numpy()
    var=df.var(ddof=1,axis=0).to_numpy()
    f=np.abs(np.fft.rfft(ts,axis=0,norm="forward"))**2
    lab=np.fft.rfftfreq(n=ts.shape[0],d=2)
    bands=[(0.01,0.04),(0.04,0.10),(0.10,0.25)]
    b=[f[np.where((lab>=lo)&(lab<hi))[0],:].sum(axis=0) for lo,hi in bands]
    return np.column_stack([mean,var]+b)

subs=sorted(glob.glob(str(RP/"data"/"Subjects"/"sub-*")))[:6]
print(f"{'subject':<10}{'condition':<22}{'FC max|diff|':>14}{'nodes max|diff|':>17}{'T':>5}")
worst_fc=worst_nd=0.0
lens=set()
for d in subs:
    sid=os.path.basename(d)
    for c in CONDS:
        tp=os.path.join(d,f"{sid}_ROITimeSeries{c}.npy")
        fp=os.path.join(d,f"{sid}_FCMatrixCondition{c}.npy")
        if not(os.path.exists(tp) and os.path.exists(fp)):
            continue
        ts=np.load(tp).astype(np.float64)
        lens.add(ts.shape[0])
        dfc=float(np.nanmax(np.abs(fc_from_ts(ts)-np.load(fp))))
        ref=[]
        dd=pd.DataFrame(ts)
        for i in range(dd.shape[1]):
            fr=np.abs(np.fft.rfft(dd.iloc[:,i],axis=0,norm="forward"))**2
            lb=np.fft.rfftfreq(n=len(dd.iloc[:,i]),d=2)
            idx=[np.where((lb>=a)&(lb<b)) for a,b in[(0.01,.04),(.04,.1),(.1,.25)]]
            ref.append([dd.iloc[:,i].mean(),dd.iloc[:,i].var(ddof=1)]+[float(fr[j[0]].sum()) for j in idx])
        dnd=float(np.max(np.abs(nodes_from_ts(ts)-np.asarray(ref))))
        worst_fc=max(worst_fc,dfc);worst_nd=max(worst_nd,dnd)
        print(f"{sid:<10}{c:<22}{dfc:>14.2e}{dnd:>17.2e}{ts.shape[0]:>5}")

print()
print(f"  worst FC deviation    {worst_fc:.3e}")
print(f"  worst node deviation  {worst_nd:.3e}")
print(f"  timeseries lengths seen: {sorted(lens)}")
print(f"  -> forward path {'reproduces exactly' if max(worst_fc,worst_nd)<1e-9 else 'does not match'}")
