import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold, StratifiedKFold

_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))

from config import config
from evaluate import cluster_evaluate

o=config.resultsRoot

def build():
    from dataset import datasetPreparation
    from gnn_encoder import GNNEncoder
    from clustering import cluster as C
    ck=torch.load(config.trainSave,map_location="cpu",weights_only=False)
    en=GNNEncoder()
    en.load_state_dict(ck["model"])
    en.eval()
    ds=datasetPreparation(fm_only=False)
    if "nodeMean" not in ck or "nodeStd" not in ck:
        raise RuntimeError("checkpoint lacks normStats")
    ds.applyNormalization(ck["nodeMean"],ck["nodeStd"])
    cl=C(en,str(config.trainSave),config.conditions,ds.subjectList)
    cl.deploy(ds.subjectData)
    x,su,co=[],[],[]
    for s,em in cl.subjectEmbeddings.items():
        e=em.detach().cpu().numpy()
        for i in range(e.shape[0]):
            x.append(e[i])
            su.append(s)
            co.append(i)
    return np.asarray(x,dtype=float),np.asarray(su),np.asarray(co,dtype=int)

def ev(x,y,k,n,t):
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    km=KMeans(n_clusters=k,n_init=config.kmeansNInit,
              random_state=config.randomSeed).fit_predict(xn)
    a=adjusted_rand_score(y,km)
    s=silhouette_score(xn,km)
    p=cluster_evaluate().perm(x,km,n_permutations=n,
                              match_selection=False,match_geometry=True)
    print(f"  {t:22s} k={k:<3d} ARI={a:+.4f} sil={s:.4f} p={p:.4f}")
    return {"variant":t,"k":int(k),"ari_vs_truth":float(a),"silhouette":float(s),
            "corrected_p":float(p),"n_items":int(len(x))}

def _raw_decoding():
    p=config.resultsRoot/"handoff_20260802"/"feature_validity"/"feature_validity.json"
    if not p.exists():
        raise FileNotFoundError(f"raw-edge decoding baseline not found at {p}. Run analysis/provenance_scripts/feature_validity.py first; this driver does not hardcode the raw accuracies.")
    fv=json.loads(p.read_text())
    return (float(fv["task condition (7-way)"]["accuracy"]),
            float(fv["subject identity"]["accuracy"]))

def sup(x,su,co):
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    ac=cross_val_score(LogisticRegression(max_iter=2000),xn,co,
                       cv=GroupKFold(n_splits=5),groups=su).mean()
    u={s:i for i,s in enumerate(sorted(set(su.tolist())))}
    yi=np.array([u[s] for s in su])
    ai=cross_val_score(LogisticRegression(max_iter=2000),xn,yi,
                       cv=StratifiedKFold(3)).mean()
    cr,ir=_raw_decoding()
    print(f"  emb: cond {ac:.4f} (ch {1/7:.4f}) ident {ai:.4f} (ch {1/len(u):.4f})")
    print(f"  raw: cond {cr:.4f} ident {ir:.4f}  (from feature_validity.json)")
    return {"condition_from_embeddings":float(ac),
            "condition_chance":1/7,
            "identity_from_embeddings":float(ai),
            "identity_chance":1/len(u),
            "condition_from_raw_edges":cr,
            "identity_from_raw_edges":ir}

def main(n=1000):
    x,su,co=build()
    ns=len(set(su.tolist()))
    print(f"{len(x)} graphs {ns} subj {len(set(co.tolist()))} cond d={x.shape[1]}")
    rows=[ev(x,co,config.nConditions,n,"raw vs condition")]
    u={s:i for i,s in enumerate(sorted(set(su.tolist())))}
    rows.append(ev(x,np.array([u[s] for s in su]),ns,n,"raw vs subject"))
    xc=x.copy()
    for s in set(su.tolist()):
        m=su==s
        xc[m]-=xc[m].mean(axis=0,keepdims=True)
    rows.append(ev(xc,co,config.nConditions,n,"within-subj vs cond"))
    sd=sup(x,su,co)
    o.mkdir(parents=True,exist_ok=True)
    d={"n_perm":n,"n_items":int(len(x)),"n_subjects":ns,
       "n_conditions":int(config.nConditions),"d":int(x.shape[1]),
       "null":"fixed-k, sphere-matched (observed k is fixed a priori, not searched)",
       "results":rows,"supervised_diagnostic":sd}
    (o/"end_to_end_positive_control.json").write_text(json.dumps(d,indent=2))
    print(f"wrote {o}/end_to_end_positive_control.json")
    return d

if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 1000)
