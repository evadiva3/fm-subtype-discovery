import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
_R=Path(__file__).resolve().parent.parent
for _p in (_R, _R/"src", _R/"models", _R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from config import config
from torch_geometric.data import Batch
from dataset import datasetPreparation
from gnn_encoder import GNNEncoder
from models.attention_pool import condition_attention_pool
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from analysis.evaluate import cluster_evaluate
ALPHAS=np.logspace(-3, 4, 20)
def fm_subs():
    ds=datasetPreparation(avgCond=False, fm_only=False)
    fm=[s for s in ds.subjectData if int(s["group_label"].view(-1)[0])==0]
    return sorted(fm, key=lambda s: s["subject_id"])
def fm_ids(fm):
    return [s["subject_id"] for s in fm]
def load_enc():
    ck=torch.load(config.trainSave, map_location="cpu")
    enc=GNNEncoder(); enc.load_state_dict(ck["model"]); enc.eval()
    pool=condition_attention_pool(d_model=config.dModel, num_cons=config.nConditions); pool.load_state_dict(ck["pool"]); pool.eval()
    return enc, pool
def rand_enc(seed=None):
    seed=config.randomSeed if seed is None else seed
    torch.manual_seed(seed)
    enc=GNNEncoder(); enc.eval()
    pool=condition_attention_pool(d_model=config.dModel, num_cons=config.nConditions); pool.eval()
    return enc, pool
def emb_attention(enc, pool, fm):
    X=[]
    with torch.no_grad():
        for s in fm:
            pc=enc(Batch.from_data_list(s["graphs"]))
            pl, _w, _t=pool(pc)
            X.append(pl.numpy())
    return np.stack(X)
def emb_mean(enc, fm):
    X=[]
    with torch.no_grad():
        for s in fm:
            pc=enc(Batch.from_data_list(s["graphs"]))
            X.append(pc.mean(0).numpy())
    return np.stack(X)
def eff_rank(X):
    ev=PCA().fit(X).explained_variance_
    er=float((ev.sum()**2)/np.sum(ev**2))
    pc1=float(ev[0]/ev.sum()*100)
    return er, pc1
def guard(label):
    n=len(label)
    mn=max(config.minClusterSizeFloor, round(config.minClusterSizeFraction*n))
    _, cnt=np.unique(label, return_counts=True)
    return bool((cnt>=mn).all())
def cluster_eval(Xraw):
    Xn=Xraw/(np.linalg.norm(Xraw, axis=1, keepdims=True)+1e-8)
    ev=cluster_evaluate()
    n=len(Xn)
    mn=max(config.minClusterSizeFloor, round(config.minClusterSizeFraction*n))
    sils=[]; labs=[]; mem=[]
    for k in config.kmeansKRange:
        lab=KMeans(n_clusters=k, n_init=config.kmeansNInit, random_state=config.randomSeed).fit_predict(Xn)
        sils.append(silhouette_score(Xn, lab)); labs.append(lab); mem.append(int(np.bincount(lab).min()))
    passed=[i for i in range(len(sils)) if mem[i]>=mn]
    if passed:
        bi=max(passed, key=lambda i: sils[i]); sizeOk=True
    else:
        bi=int(np.argmax(sils)); sizeOk=False
    k=config.kmeansKRange[bi]; s=sils[bi]; lab=labs[bi]
    gap=ev.gap_stat(Xn, list(config.kmeansKRange))
    kgap=ev.gap_k(gap, config.kmeansKRange)
    pp=ev.perm(Xn, lab)
    er, pc1=eff_rank(Xraw)
    return {"silhouette": float(s), "k_sel": int(k), "perm_p": float(pp), "sizeOk": sizeOk,
            "k_gap": float(kgap), "eff_rank": er, "pc1_pct": pc1, "rank_ceiling": min(Xraw.shape)-1, "label": lab}
def severity_y(ids):
    clin=pd.read_csv(config.clinicalCsv); clin["subject_id"]=clin["subject_id"].astype(str)
    df=clin.set_index("subject_id").loc[list(ids)]
    cols=["vas_pain", "hamd_total", "hama_total"]
    z=(df[cols]-df[cols].mean())/df[cols].std(ddof=1)
    return z.mean(axis=1).to_numpy()

def _folds(X):
    out=[]
    for tr, te in LeaveOneOut().split(X):
        sc=StandardScaler().fit(X[tr])
        out.append((tr, te, sc.transform(X[tr]), sc.transform(X[te])))
    return out

def loo_predict(fit_predict, X, y):
    preds=np.zeros(len(y))
    for tr, te, Xtr, Xte in _folds(X):
        preds[te]=fit_predict(Xtr, y[tr], Xte)
    return preds
def _ridge(Xtr, ytr, Xte):
    return RidgeCV(alphas=ALPHAS).fit(Xtr, ytr).predict(Xte)
def score_r(preds, y):
    r=float(np.corrcoef(preds, y)[0, 1])
    r2=float(1.0-((y-preds)**2).sum()/((y-y.mean())**2).sum())
    return r, r2
def perm_r(fit_predict, X, y, r_obs, seed=None):
    seed=config.randomSeed if seed is None else seed
    rng=np.random.default_rng(seed)
    c=0
    for _ in range(config.nPermutations):
        yp=rng.permutation(y)
        pr=loo_predict(fit_predict, X, yp)
        if np.corrcoef(pr, yp)[0, 1]>r_obs:
            c+=1
    return c/config.nPermutations
