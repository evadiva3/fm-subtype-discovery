from __future__ import annotations
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))

from config import config
from evaluate import cluster_evaluate

o=config.resultsRoot
K=int(os.environ.get("E2E_K",7))
ND=int(os.environ.get("NDATASETS",200))
NR=int(os.environ.get("NDRAWS",200))
EMB=_R/"data"/"outputs"/"e2e_graph_embeddings.npy"

def _pr(x):
    e=np.linalg.eigvalsh(np.cov(x,rowvar=False))
    e=e[e>1e-12]
    return float((e.sum()**2)/(e**2).sum())

def _wilson(k,n,z=1.959963985):
    p=k/n
    dn=1+z*z/n
    c=(p+z*z/(2*n))/dn
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/dn
    return float(max(0.0,c-h)),float(min(1.0,c+h))

def _one(a):
    i,xn=a
    rng=np.random.default_rng(900_000+i)
    ev=cluster_evaluate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sy=ev._null_mvn(xn,rng)
        lab=KMeans(n_clusters=K,n_init=config.kmeansNInit,random_state=i).fit_predict(sy)
        p=ev.perm(sy,lab,n_permutations=NR,random_state=i,match_selection=False,match_geometry=True)
    return float(p)

def main():
    if not EMB.exists():
        raise FileNotFoundError(
            f"{EMB} not found. Generate it with analysis/end_to_end_positive_control.py, "
            f"which caches the 406 graph embeddings it builds.")
    x=np.load(EMB).astype(float)
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    print(f"e2e calibration: n={xn.shape[0]} d={xn.shape[1]} participation ratio {_pr(xn):.2f}")
    print(f"  fixed k={K}, {ND} structureless datasets x {NR} draws")

    with mp.Pool(min(ND,os.cpu_count() or 4)) as pool:
        pv=np.array(pool.map(_one,[(i,xn) for i in range(ND)]))

    rej=float(np.mean(pv<0.05))
    lo,hi=_wilson(int((pv<0.05).sum()),ND)
    from scipy import stats
    ks=stats.kstest(pv,"uniform")
    d={"n":int(xn.shape[0]),"d":int(xn.shape[1]),"k":K,
       "participation_ratio":_pr(xn),"n_datasets":ND,"n_draws":NR,
       "mean_p":float(pv.mean()),"median_p":float(np.median(pv)),
       "rejection_rate_05":rej,"rejection_rate_05_ci95":[lo,hi],
       "ks_D":float(ks.statistic),"ks_p":float(ks.pvalue)}
    o.mkdir(parents=True,exist_ok=True)
    (o/"e2e_calibration.json").write_text(json.dumps(d,indent=2))
    np.save(o/"e2e_calibration_pvalues.npy",pv)

    print(f"  mean p {pv.mean():.4f}   (nominal 0.500)")
    print(f"  rejection @0.05 {rej:.4f}  [{lo:.4f}, {hi:.4f}]   (nominal 0.050)")
    print(f"  KS vs U(0,1)    D={ks.statistic:.4f}, p={ks.pvalue:.3e}")
    print(f"  wrote {o}/e2e_calibration.json")
    return d

if __name__=="__main__":
    main()
