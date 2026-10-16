from __future__ import annotations
import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
from _paths import RP,RUNS,CAN as CANON,out
import json,time
import numpy as np,pandas as pd

OUT=out("audit_gap_fills")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from analysis.evaluate import cluster_evaluate
from config import config

OUT.mkdir(parents=True,exist_ok=True)
ND=int(os.environ.get("NDRAWS",20000))
CAN=CANON
emb=np.load(CAN/"Embeddings.npy").astype(float)
lab=pd.read_csv(CAN/"K-Means-Labeling.csv").sort_values("Subject_Id")["Label"].to_numpy()
ev=cluster_evaluate()
k=len(np.unique(lab))
KR=list(config.kmeansKRange)
MS=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*emb.shape[0]))
embN=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-8)

def ambient(x,rng):
    return rng.multivariate_normal(x.mean(0),np.cov(x,rowvar=False),size=x.shape[0],method="svd")

def best_sel(d,rs):
    s,m=[],[]
    for kk in KR:
        lb=KMeans(n_clusters=kk,n_init=config.kmeansNInit,random_state=rs).fit_predict(d)
        s.append(silhouette_score(d,lb));m.append(int(np.bincount(lb).min()))
    ok=[j for j in range(len(KR)) if m[j]>=MS]
    return s[max(ok,key=lambda j:s[j]) if ok else int(np.argmax(s))]

print(f"canonical checkpoint {emb.shape}, selected k={k}, guard={MS}, {ND:,} draws")
obs_amb=silhouette_score(emb,lab)
obs_sph=silhouette_score(embN,lab)
print(f"  observed silhouette: ambient {obs_amb:.4f}   normalized {obs_sph:.4f}\n")

rows=[]
rng=np.random.default_rng(config.randomSeed);c=0;nm=[]
mu,cov=embN.mean(0),np.cov(embN,rowvar=False)
t=time.perf_counter()
for i in range(ND):
    nl=rng.multivariate_normal(mu,cov,size=embN.shape[0],method="svd")
    s=silhouette_score(nl,KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=i).fit_predict(nl))
    nm.append(s);c+=s>=obs_sph
rows.append(("1. misspecified (no sphere projection)",(c+1)/(ND+1),float(np.mean(nm)),time.perf_counter()-t))

rng=np.random.default_rng(config.randomSeed);c=0;nm=[]
t=time.perf_counter()
for i in range(ND):
    nl=ev._null_mvn(embN,rng)
    s=silhouette_score(nl,KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=i).fit_predict(nl))
    nm.append(s);c+=s>=obs_sph
rows.append(("2. geometry corrected only",(c+1)/(ND+1),float(np.mean(nm)),time.perf_counter()-t))

rng=np.random.default_rng(config.randomSeed);c=0;nm=[]
t=time.perf_counter()
for i in range(ND):
    nl=ev._null_mvn(embN,rng)
    s=best_sel(nl,i)
    nm.append(s);c+=s>=obs_sph
rows.append(("3. fully corrected",(c+1)/(ND+1),float(np.mean(nm)),time.perf_counter()-t))

print(f"  {'construction':<36}{'p':>10}{'null mean sil':>16}{'min':>8}")
for nm_,p,ns,el in rows:
    print(f"  {nm_:<36}{p:>10.4f}{ns:>16.4f}{el/60:>8.1f}")
json.dump({"n_draws":ND,"observed_sil_ambient":float(obs_amb),
           "observed_sil_normalized":float(obs_sph),"k":int(k),
           "progression":[{"construction":a,"p":b,"null_mean_sil":c_}
                          for a,b,c_,_ in rows]},
          open(OUT/"three_way_null_canonical.json","w"),indent=2)

print("\ntodo 2.5: fm bootstrap-vs-cross-run ratio, 14-run ari")
cr=json.load(open((RUNS/"crossrun_ari_summary.json")))["mean_ari"]
bp=RP/"results/bootstrap_stability.csv"
if bp.exists():
    bs=pd.read_csv(bp)
    ac=[c for c in bs.columns if "ari" in c.lower()]
    if ac:
        bm=float(bs[ac[0]].mean())
        print(f"  bootstrap ARI (mean of {len(bs)} resamples, col '{ac[0]}')  {bm:.4f}")
        print(f"  cross-run ARI (14 corrected-search runs, 91 pairs)      {cr:.4f}")
        print(f"  ratio {bm/cr:.2f}x   (paper cites 2.5x for FM from the superseded runs;"
              f" MDD is 1.6x)")
        json.dump({"bootstrap_ari":bm,"crossrun_ari":cr,"ratio":bm/cr},
                  open(OUT/"bootstrap_vs_crossrun.json","w"),indent=2)
    else:
        print(f"  no ARI column in bootstrap_stability.csv; columns: {list(bs.columns)}")
else:
    print("  results/bootstrap_stability.csv not found")
print(f"\n  wrote {OUT}")
