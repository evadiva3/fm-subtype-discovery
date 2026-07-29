from __future__ import annotations
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rp=Path(__file__).resolve().parents[1] 
sys.path.insert(0,str(rp))
from analysis.evaluate import cluster_evaluate
from config import config
o=rp/"results"/"calibration"
nd=int(os.environ.get("NDATASETS",200))
nr=int(os.environ.get("NDRAWS",200))
def selLab(d,rs=0):
    n=d.shape[0]
    ms=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*n))
    s,m,l=[],[],[]
    for k in config.kmeansKRange:
        lb=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=rs).fit_predict(d)
        s.append(silhouette_score(d,lb))
        m.append(int(np.bincount(lb).min()))
        l.append(lb)
    p=[i for i in range(len(config.kmeansKRange)) if m[i]>=ms]
    b=max(p,key=lambda i:s[i]) if p else int(np.argmax(s))
    return l[b]
def oneDs(a):
    i,xn=a
    rng=np.random.default_rng(10_000+i)
    ev=cluster_evaluate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sy=ev._null_mvn(xn,rng)
        lb=selLab(sy,rs=i)
        p=ev.perm(sy,lb,n_permutations=nr,random_state=i,match_selection=True)
    return float(p),int(len(np.unique(lb)))
def main()->None:
    x=np.load(rp/"data/outputs/trained_fm_embeddings.npy").astype(float)
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    print(f"calibration: {nd} datasets x {nr} draws, embeddings {x.shape}")
    with mp.Pool(min(nd,os.cpu_count() or 4)) as pool:
        res=pool.map(oneDs,[(i,xn) for i in range(nd)])
    pv=np.array([r[0] for r in res])
    ks=np.array([r[1] for r in res])
    rj=float(np.mean(pv<0.05))
    from scipy import stats
    kst=stats.kstest(pv,"uniform")
    o.mkdir(parents=True,exist_ok=True)
    np.save(o/"calibration_pvalues.npy",pv)
    sm={
        "n_datasets":nd,"n_draws":nr,
        "mean_p":float(pv.mean()),"median_p":float(np.median(pv)),
        "rejection_rate_05":rj,
        "ks_D":float(kst.statistic),"ks_p":float(kst.pvalue),
        "selected_k_counts":{int(k):int((ks==k).sum()) for k in np.unique(ks)},
        "paper_reports":{"mean_p":0.614,"rejection_rate_05":0.005,
                          "ks_D":0.188,"ks_p":"<1e-6"},
    }
    (o/"calibration_summary.json").write_text(json.dumps(sm,indent=2))

    print(f"  mean p           {pv.mean():.4f}   (paper 0.614, nominal 0.500)")
    print(f"  median p         {np.median(pv):.4f}")
    print(f"  rejection @0.05  {rj:.4f}   (paper 0.005, nominal 0.050)")
    print(f"  KS vs U(0,1)     D = {kst.statistic:.4f}, p = {kst.pvalue:.3e}"
          f"   (paper D = 0.188)")
    print(f"  wrote {o}/calibration_pvalues.npy")


if __name__=="__main__":
    main()
