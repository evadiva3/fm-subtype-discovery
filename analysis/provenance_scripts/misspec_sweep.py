from __future__ import annotations
import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
from _paths import CAN as CANON,out
import itertools
import numpy as np,pandas as pd

OUT=out("audit_gap_fills")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import config

ND=int(os.environ.get("NDRAWS",5000))
TARGET=0.1685
CAN=CANON
emb=np.load(CAN/"Embeddings.npy").astype(float)
lab=pd.read_csv(CAN/"K-Means-Labeling.csv").sort_values("Subject_Id")["Label"].to_numpy()
k=len(np.unique(lab))
embN=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-8)
obs={"ambient":silhouette_score(emb,lab),"normalized":silhouette_score(embN,lab)}
src={"ambient":emb,"normalized":embN}
print(f"canonical {emb.shape}, k={k}, {ND:,} draws, target p={TARGET}")
print(f"  observed: ambient {obs['ambient']:.4f}  normalized {obs['normalized']:.4f}\n")

rows=[]
for fit,proj in itertools.product(("ambient","normalized"),(False,True)):
    x=src[fit]
    rng=np.random.default_rng(config.randomSeed)
    mu,cov=x.mean(0),np.cov(x,rowvar=False)
    sil=np.empty(ND)
    for i in range(ND):
        nl=rng.multivariate_normal(mu,cov,size=x.shape[0],method="svd")
        if proj:
            nl=nl/(np.linalg.norm(nl,axis=1,keepdims=True)+1e-8)
        sil[i]=silhouette_score(nl,KMeans(n_clusters=k,n_init=config.kmeansNInit,
                                          random_state=i).fit_predict(nl))
    for stat in("ambient","normalized"):
        p=(int((sil>=obs[stat]).sum())+1)/(ND+1)
        rows.append({"fitSpace":fit,"projDraws":proj,"statSpace":stat,
                     "null_mean_sil":float(sil.mean()),"p":p,
                     "delta_to_target":abs(p-TARGET)})

df=pd.DataFrame(rows).sort_values("delta_to_target")
print(f"  {'fit':>11}{'project':>9}{'stat':>12}{'null mean':>11}{'p':>9}{'|p-0.1685|':>12}")
for _,r in df.iterrows():
    print(f"  {r['fitSpace']:>11}{str(r['projDraws']):>9}{r['statSpace']:>12}"
          f"{r['null_mean_sil']:>11.4f}{r['p']:>9.4f}{r['delta_to_target']:>12.4f}")
b=df.iloc[0]
print(f"\n  closest: fit={b['fitSpace']}, project={b['projDraws']}, stat={b['statSpace']}"
      f"  ->  p={b['p']:.4f}")
print("  match" if b["delta_to_target"]<0.03 else
      "  no variant reproduces 0.1685 -- likely a different checkpoint or code revision")
df.to_csv(OUT/"misspec_sweep.csv",index=False)
print(f"  wrote {OUT/'misspec_sweep.csv'}")
