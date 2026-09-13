from __future__ import annotations
import os,sys,json,time,warnings,gc,io,contextlib
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS","3")

RP=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",".."))
for sub in("","src","models","preprocessing",os.path.join("experiments","ds002748_mdd")):
    p=os.path.join(RP,sub) if sub else RP
    if p not in sys.path:
        sys.path.insert(0,p)

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import silhouette_score
from config import config

_P=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"mdd_params.json")))
for _k,_v in _P.items():
    setattr(config,_k,_v)

import subject_filter_mdd as _sf
def _subs():
    return sorted(f.name for f in _sf.ROOT.iterdir()
                  if f.is_dir() and (_sf.ROOT/_sf.TS_PAT.format(s=f.name)).exists())
_sf.get_included_subjects_mdd=_subs

import dataset_mdd as _dm

ROOT=Path(os.environ.get("MDD_CTRL_ROOT",os.path.join(RP,"results","mdd","fixed_arch_control","_run")))
CK=ROOT/"ckpt"
RES=ROOT/"runs"
for d in(CK,RES):
    d.mkdir(parents=True,exist_ok=True)


def run(sd):
    import importlib
    import dataset_mdd as dsm
    importlib.reload(dsm)

    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from augmentations import graph_augmentor
    from train_mdd import train_mdd
    from clustering_mdd import mddCluster
    from torch.utils.data import DataLoader,random_split

    config.randomSeed=sd
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)

    ds=dsm.mddDataset()

    class W(torch.utils.data.Dataset):
        def __init__(self,sd):self.sd=sd
        def __len__(self):return len(self.sd)
        def __getitem__(self,i):return self.sd[i]

    gd=W(ds.subjectData)
    n=len(gd)
    nv=int(n*config.valFraction)
    nt=n-nv
    gen=torch.Generator().manual_seed(sd)
    tr,va=random_split(gd,[nt,nv],generator=gen)
    ds.normalize(tr.indices)
    ns=(ds.mu,ds.sig)
    bs=max(2,nt//4)
    trL=DataLoader(tr,batch_size=bs,shuffle=True,collate_fn=lambda b:b,drop_last=True)
    vaL=DataLoader(va,batch_size=bs,shuffle=False,collate_fn=lambda b:b)

    enc=GNNEncoder().to(config.device)
    with contextlib.redirect_stdout(io.StringIO()):
        enc,_,_=train_mdd(enc,NTXentLoss(),trL,vaL,graph_augmentor(),config.device,str(CK),normStats=ns)

    cl=mddCluster(enc,str(CK),[],ds.subjectList)
    cl.deploy(ds.subjectData)
    cl._split_fm_hc()
    t,ids=cl._stack(cl.fmEmbed)
    config.randomSeed=42
    tbl=cl.KMeansUse(t,ids,skip_perm=True,skip_gap=True)
    k=int(tbl[0]["k_selected_silhouette"].iloc[0])
    sil=float(tbl[0].loc[tbl[0]["k"]==k,"silhouette_score"].iloc[0])
    lab=[int(x) for x in tbl[2]]

    aid=list(cl.attentionEmbeddings.keys())
    E=torch.stack([cl.attentionEmbeddings[i] for i in aid]).detach().cpu().numpy()
    gl=np.array([cl.groupLabels[i] for i in aid])
    E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
    sep=float(silhouette_score(E,gl))

    emb=t.detach().cpu().numpy() if hasattr(t,"detach") else np.asarray(t)
    np.save(RES/f"emb{sd:03d}.npy",emb)

    del enc,ds,cl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(seed=sd,k_selected=k,silhouette=sil,sep_sil=sep,subject_ids=list(ids),labels=lab)


def main():
    lo=int(sys.argv[1]) if len(sys.argv)>1 else 0
    hi=int(sys.argv[2]) if len(sys.argv)>2 else 20
    print(f"seeds [{lo},{hi}) device={config.device} dModel={config.dModel} heads={config.heads} layers={config.layers}",flush=True)
    for s in range(lo,hi):
        q=RES/f"seed{s:03d}.json"
        if q.exists():
            continue
        t=time.perf_counter()
        r=run(s)
        r["elapsed_s"]=round(time.perf_counter()-t,1)
        json.dump(r,open(q,"w"))
        print(f"  {s:3d} k={r['k_selected']} sil={r['silhouette']:.4f} sep={r['sep_sil']:+.4f} [{r['elapsed_s']:.0f}s]",flush=True)


if __name__=="__main__":
    main()
