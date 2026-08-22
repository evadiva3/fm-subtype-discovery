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
o=config.resultsRoot/"calibration"
nd=int(os.environ.get("NDATASETS",200))
nr=int(os.environ.get("NDRAWS",200))
sel=os.environ.get("MATCH_SELECTION","1") not in ("0","false","False")
geo=os.environ.get("MATCH_GEOMETRY","1") not in ("0","false","False")
tag={(True,True):"corrected",(False,True):"fixedk",(True,False):"ambient",
     (False,False):"misspecified"}[(sel,geo)]
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
        p=ev.perm(sy,lb,n_permutations=nr,random_state=i,
                  match_selection=sel,match_geometry=geo)
    return float(p),int(len(np.unique(lb)))
def main()->None:
    x=np.load(rp/"data/outputs/trained_fm_embeddings.npy").astype(float)
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    print(f"calibration [{tag}]: {nd} datasets x {nr} draws, embeddings {x.shape}"
          f"  (sel={sel}, geo={geo})")
    with mp.Pool(min(nd,os.cpu_count() or 4)) as pool:
        res=pool.map(oneDs,[(i,xn) for i in range(nd)])
    pv=np.array([r[0] for r in res])
    ks=np.array([r[1] for r in res])
    rj=float(np.mean(pv<0.05))
    from scipy import stats
    kst=stats.kstest(pv,"uniform")
    z=1.959963985
    ns=int(round(rj*nd))
    den=1.0+z*z/nd
    ctr=(rj+z*z/(2*nd))/den
    hw=z*np.sqrt(rj*(1-rj)/nd+z*z/(4*nd*nd))/den
    lo,hi=max(0.0,ctr-hw),min(1.0,ctr+hw)
    o.mkdir(parents=True,exist_ok=True)
    np.save(o/f"calibration_pvalues_{tag}.npy",pv)
    sm={
        "construction":tag,"match_selection":sel,"match_geometry":geo,
        "n_datasets":nd,"n_draws":nr,
        "mean_p":float(pv.mean()),"median_p":float(np.median(pv)),
        "rejection_rate_05":rj,
        "n_rejections":ns,
        "rejection_rate_05_ci95":[float(lo),float(hi)],
        "ks_D":float(kst.statistic),"ks_p":float(kst.pvalue),
        "selected_k_counts":{int(k):int((ks==k).sum()) for k in np.unique(ks)},
    }
    (o/f"calibration_summary_{tag}.json").write_text(json.dumps(sm,indent=2))
    if tag=="corrected":
        np.save(o/"calibration_pvalues.npy",pv)
        (o/"calibration_summary.json").write_text(json.dumps(sm,indent=2))

    print(f"  mean p           {pv.mean():.4f}   (nominal 0.500)")
    print(f"  median p         {np.median(pv):.4f}")
    print(f"  rejection @0.05  {rj:.4f}  [{lo:.4f}, {hi:.4f}] 95% CI, {ns}/{nd}"
          f"   (nominal 0.050)")
    print(f"  KS vs U(0,1)     D = {kst.statistic:.4f}, p = {kst.pvalue:.3e}")
    print(f"  wrote {o}/calibration_pvalues_{tag}.npy")
    print(f"  wrote {o}/calibration_summary_{tag}.json")
    if tag=="corrected":
        print("  also refreshed the unsuffixed corrected filenames")


if __name__=="__main__":
    main()
