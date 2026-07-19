#experimental ignore for now


import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_R=Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent,_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
from config import config
from clustering_mdd import mddCluster

OUT=Path(config.resultsRoot)/"mdd"

def _sil(tr):
    d=tr[0]
    ks=int(d["k_selected_silhouette"].iloc[0])
    return float(d.loc[d["k"]==ks,"silhouette_score"].iloc[0])

def rand_encoder(subs,ids):
    from gnn_encoder import GNNEncoder
    rows=[]
    for sd in range(10):
        torch.manual_seed(sd)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sd)
        e=GNNEncoder()
        r=mddCluster(e,str(config.checkpointDir),None,ids)
        r.deploy(subs)
        r._split_fm_hc()
        t,tids=r._stack(r.fmEmbed)
        tr=r.KMeansUse(t,tids,skip_perm=True,skip_gap=True)
        rows.append({"seed":sd,"silhouette":_sil(tr)})
    return pd.DataFrame(rows)

def trained(subs,ids):
    from gnn_encoder import GNNEncoder
    ck=torch.load(config.checkpointDir/"bestMddModel.pt",map_location="cpu")
    e=GNNEncoder()
    e.load_state_dict(ck["enc"])
    r=mddCluster(e,str(config.checkpointDir),None,ids)
    r.deploy(subs)
    r._split_fm_hc()
    t,tids=r._stack(r.fmEmbed)
    return _sil(r.KMeansUse(t,tids,skip_perm=True,skip_gap=True))

if __name__=="__main__":
    from dataset_mdd import mddDataset
    ds=mddDataset()
    rdf=rand_encoder(ds.subjectData,ds.subjectList)
    OUT.mkdir(parents=True,exist_ok=True)
    rdf.to_csv(OUT/"random_encoder_ablation.csv",index=False)
    print(rdf.to_string(index=False))
    print(f"random seeds 0-9: silhouette mean={rdf['silhouette'].mean():.4f} range=[{rdf['silhouette'].min():.4f},{rdf['silhouette'].max():.4f}]")
    ts=trained(ds.subjectData,ds.subjectList)
    print(f"trained encoder silhouette={ts:.4f}")
