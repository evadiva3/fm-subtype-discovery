import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")
_R=Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent,_R,_R/"src",_R/"models",_R/"analysis",_R/"preprocessing"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
from config import config
from torch_geometric.data import Batch
from dataset_mdd import mddDataset,COLUMN_MAP,PART_TSV
from gnn_encoder import GNNEncoder
from severity_gradient_regression import make_folds,loo_predictions,score,permutation_p
SEED=config.randomSeed
SEV_PRIMARY="zsrds"
SEV_SENS=["bdi","madrs"]
def embed(enc,subs,ids):
    enc.eval()
    d={}
    with torch.no_grad():
        for s in subs:
            b=Batch.from_data_list(s["graphs"])
            d[s["subject_id"]]=enc(b).view(-1)
    return np.stack([d[i].detach().cpu().numpy() for i in ids])
def severity(pf,ids,key):
    s=pd.to_numeric(pf.set_index(COLUMN_MAP["subject_id"])[COLUMN_MAP[key]],errors="coerce")
    return s.reindex(ids)
def run_space(nm,X,y):
    fo=make_folds(X)
    pr=loo_predictions(fo,y)
    r,r2=score(pr,y)
    p,_=permutation_p(fo,y,r)
    print(f"{nm}: n={len(y)} dim={X.shape[1]} r={r:+.4f} R2={r2:+.4f} perm_p={p:.4f}")
    return {"space":nm,"n":len(y),"dim":X.shape[1],"r":r,"R2":r2,"perm_p":p}
def rand_embed(sd,subs,ids):
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)
    return embed(GNNEncoder(),subs,ids)
def run_scale(tag,key,enc,subs,pf):
    ids0=sorted(s["subject_id"] for s in subs if int(s["group_label"].view(-1)[0].item())==0)
    yr=severity(pf,ids0,key)
    ids=[i for i in ids0 if pd.notna(yr[i])]
    y=yr.loc[ids].to_numpy(dtype=float)
    y=(y-y.mean())/y.std(ddof=1)
    xm=embed(enc,subs,ids)
    res=[run_space(f"{tag}-main-trained",xm,y),run_space(f"{tag}-random-control-seed{SEED}",rand_embed(SEED,subs,ids),y)]
    rrows=[]
    for sd in range(10):
        r=run_space(f"{tag}-random-seed-{sd}",rand_embed(sd,subs,ids),y)
        rrows.append({"seed":sd,"r":r["r"],"R2":r["R2"],"perm_p":r["perm_p"]})
    out=Path(config.resultsRoot)/"mdd"
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(res).to_csv(out/f"severity_regression_mdd_{tag}.csv",index=False)
    pd.DataFrame(rrows).to_csv(out/f"severity_random_seeds_mdd_{tag}.csv",index=False)
    print(f"[{tag}] scale={COLUMN_MAP[key]} n={len(ids)}")
    print(pd.DataFrame(res).to_string(index=False))
    return res
def main():
    ds=mddDataset()
    subs=ds.subjectData
    pf=pd.read_csv(PART_TSV,sep="\t")
    pf[COLUMN_MAP["subject_id"]]=pf[COLUMN_MAP["subject_id"]].astype(str)
    ck=torch.load(config.checkpointDir/"bestMddModel.pt",map_location="cpu")
    enc=GNNEncoder()
    enc.load_state_dict(ck["enc"])
    print("PRIMARY: Zung_SDS")
    run_scale(SEV_PRIMARY,SEV_PRIMARY,enc,subs,pf)
    for k in SEV_SENS:
        print(f"SENSITIVITY: {COLUMN_MAP[k]}")
        run_scale(k,k,enc,subs,pf)

if __name__=="__main__":
    main()
