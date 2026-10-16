from __future__ import annotations
import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
from _paths import CAN as CANON,out
import json
import numpy as np,pandas as pd

OUT=out("audit_gap_fills")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import config

ND=int(os.environ.get("NDRAWS",20000))
CAN=CANON
emb=np.load(CAN/"Embeddings.npy").astype(float)
lab=pd.read_csv(CAN/"K-Means-Labeling.csv").sort_values("Subject_Id")["Label"].to_numpy()
k=len(np.unique(lab))
embN=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-8)
obs_amb=silhouette_score(emb,lab)
obs_sph=silhouette_score(embN,lab)
print(f"canonical {emb.shape}, k={k}, {ND:,} draws")
print(f"  observed silhouette: ambient {obs_amb:.4f}   normalized {obs_sph:.4f}")

rng=np.random.default_rng(config.randomSeed)
mu,cov=emb.mean(0),np.cov(emb,rowvar=False)
sil=np.empty(ND)
for i in range(ND):
    nl=rng.multivariate_normal(mu,cov,size=emb.shape[0],method="svd")
    sil[i]=silhouette_score(nl,KMeans(n_clusters=k,n_init=config.kmeansNInit,
                                       random_state=i).fit_predict(nl))
pA=(int((sil>=obs_amb).sum())+1)/(ND+1)
pB=(int((sil>=obs_sph).sum())+1)/(ND+1)
print(f"\n  ambient null: mean sil {sil.mean():.4f}, sd {sil.std(ddof=1):.4f}")
print(f"  A. matched-ambient  (obs {obs_amb:.4f} vs ambient null)   p={pA:.4f}")
print(f"  B. mismatched       (obs {obs_sph:.4f} vs ambient null)   p={pB:.4f}")
print(f"\n  paper's misspecified value is 0.1685 -> "
      f"{'B reproduces it' if abs(pB-0.1685)<0.03 else 'neither matches; investigate'}")
np.save(OUT/"ambient_null_sil.npy",sil)
json.dump({"n_draws":ND,"obs_ambient":float(obs_amb),"obs_normalized":float(obs_sph),
           "ambient_null_mean":float(sil.mean()),"ambient_null_sd":float(sil.std(ddof=1)),
           "p_matched_ambient":pA,"p_mismatched":pB,"paper_value":0.1685},
          open(OUT/"misspec_variants.json","w"),indent=2)
print(f"  wrote {OUT}")
