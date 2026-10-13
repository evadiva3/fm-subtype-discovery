#Exploratory fitted-reference sensitivity
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
o=config.resultsRoot/"null_guarded_v7"
MODES=[("ambient_fixed_guarded",False,False), ("projected_fixed_guarded",True,False), ("ambient_searched",False,True), ("projected_searched",True,True)]

def main(n=None):
    n=int(n) if n else config.nPermutations
    x=np.load(_R/"data"/"outputs"/"trained_fm_embeddings.npy").astype(float)
    lb=pd.read_csv(_R/"data"/"outputs"/"K-Means-Labeling.csv").sort_values("Subject_Id")["Label"].to_numpy()
    ev=cluster_evaluate()
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-08)
    r=ev.silhouette(xn,lb)
    print(f"{x.shape[0]}x{x.shape[1]} k={len(np.unique(lb))} sil={r:.6f} n={n}")
    sbest,kbest=ev._select_best_silhouette(xn,random_state=0,return_k=True)
    rows,kc=({}, {})
    for t,mg,ms in MODES:
        p,k=ev.perm(x,lb,n_permutations=n,match_geometry=mg,match_selection=ms,return_k_counts=True)
        rows[t]=float(p)
        kc[t]=k
        print(f"  {t:13s} geo={str(mg):5s} sel={str(ms):5s} p={p:.4f}")
    tot=sum(kc["projected_searched"].values())
    pct={int(a):round(100.0*b/tot,2) for a,b in kc["projected_searched"].items()}
    print(f"null selected k: {pct}")
    o.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"construction":t,"geometry_matched":mg,"selection_matched":ms,"observed_silhouette":float(sbest if ms else r),"p":rows[t],"n_perm":n} for t,mg,ms in MODES]).to_csv(o/f"null_progression_{n}.csv",index=False)
    d={"n_perm":n,"observed_silhouette":float(r),"observed_k":int(len(np.unique(lb))),"searched_observed_silhouette":float(sbest),"searched_observed_k":kbest,"guard_failed_k_sentinel":0,"p_by_construction":rows,"null_selected_k_counts": {t: {int(a):int(b) for a,b in c.items()} for t,c in kc.items()},"null_selected_k_pct_projected_searched":pct}
    (o/f"null_progression_{n}.json").write_text(json.dumps(d,indent=2))
    print(f"wrote {o}/null_progression_{n}.json")
    return d
if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else None)
