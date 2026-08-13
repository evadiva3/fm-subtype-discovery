import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))

from config import config
from evaluate import cluster_evaluate

o=_R/"results"/"null_corrected"

MODES=[("misspecified",False,False),("geometry",True,False),
       ("selection",False,True),("corrected",True,True)]

def main(n=None):
    n=int(n) if n else config.nPermutations
    x=np.load(_R/"data"/"outputs"/"trained_fm_embeddings.npy").astype(float)
    lb=pd.read_csv(_R/"data"/"outputs"/"K-Means-Labeling.csv")["Label"].to_numpy()
    ev=cluster_evaluate()
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    r=ev.silhouette(xn,lb)
    print(f"{x.shape[0]}x{x.shape[1]} k={len(np.unique(lb))} sil={r:.6f} n={n}")
    rows,kc={},{}
    for t,mg,ms in MODES:
        p,k=ev.perm(x,lb,n_permutations=n,match_geometry=mg,match_selection=ms,
                    return_k_counts=True)
        rows[t]=float(p)
        kc[t]=k
        print(f"  {t:13s} geo={str(mg):5s} sel={str(ms):5s} p={p:.4f}")
    tot=sum(kc["corrected"].values())
    pct={int(a):round(100.0*b/tot,2) for a,b in kc["corrected"].items()}
    print(f"null selected k: {pct}")
    o.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"construction":t,"geometry_matched":mg,"selection_matched":ms,
                   "observed_silhouette":float(r),"p":rows[t],"n_perm":n}
                  for t,mg,ms in MODES]).to_csv(
        o/f"null_progression_{n}.csv",index=False)
    d={"n_perm":n,"observed_silhouette":float(r),"observed_k":int(len(np.unique(lb))),
       "p_by_construction":rows,
       "null_selected_k_counts":{t:{int(a):int(b) for a,b in c.items()}
                                 for t,c in kc.items()},
       "null_selected_k_pct_corrected":pct}
    (o/f"null_progression_{n}.json").write_text(json.dumps(d,indent=2))
    print(f"wrote {o}/null_progression_{n}.json")
    return d

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else None)
