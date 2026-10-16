from __future__ import annotations
import os, sys, json, time, warnings, gc
from _paths import RP, out
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS","2")

for sub in("src","models","preprocessing"):
    sys.path.insert(0,os.path.join(str(RP),sub))

import numpy as np
import torch
from pathlib import Path
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from config import config
from subject_filter import get_included_subjects

ROOT=out("propagated_null")
WORK=ROOT/"work"
SURR=WORK/"Subjects"
CKPT=WORK/"ckpt"
RESULTS=ROOT/"draws"
for d in(SURR,CKPT,RESULTS):
    d.mkdir(parents=True,exist_ok=True)

REAL=Path(config.subjectDataFolder)
CONDS=[c.replace(" ","") for c in config.conditions]
SUBS=get_included_subjects()
TMOD=48

def fit_group_model():
    model,lengths={},{}
    for c in CONDS:
        X,ln=[],{}
        for s in SUBS:
            p=REAL/s/f"{s}_ROITimeSeries{c}.npy"
            ts=np.load(p).astype(np.float64)
            ln[s]=ts.shape[0]
            if ts.shape[0]==TMOD:
                X.append(ts)
        X=np.asarray(X)
        Xf=np.fft.rfft(X,axis=1)
        n,F,P=Xf.shape
        As=[]
        for f in range(F):
            z=Xf[:,f,:]
            S=(z.conj().T@z)/n
            w,V=np.linalg.eigh(S)
            keep=w>max(w.max(),0)*1e-12
            As.append((V[:,keep]*np.sqrt(w[keep])).astype(np.complex128))
        model[c]=As
        lengths[c]=ln
    return model,lengths,F

def draw_subject(As,F,rng):
    P=As[0].shape[0]
    Z=np.zeros((F,P),dtype=np.complex128)
    for f in range(F):
        A=As[f]; r=A.shape[1]
        if r==0:
            continue
        if f==0 or f==F-1:  
            Z[f]=A@rng.standard_normal(r)
        else:
            w=(rng.standard_normal(r)+1j*rng.standard_normal(r))/np.sqrt(2.0)
            Z[f]=A@w
    return np.fft.irfft(Z,n=TMOD,axis=0)

FCM=ConnectivityMeasure(kind="correlation",cov_estimator=LedoitWolf())

def fc_of(ts):
    z=np.arctanh(FCM.fit_transform([ts])[0])
    np.fill_diagonal(z,0)
    z[~np.isfinite(z)]=0
    return z.astype(np.float32)

def materialise(model,lengths,F,seed):
    rng=np.random.default_rng(seed)
    for s in SUBS:
        d=SURR/s
        d.mkdir(parents=True,exist_ok=True)
        for c in CONDS:
            ts=draw_subject(model[c],F,rng)
            T=lengths[c].get(s,TMOD)
            if T!=TMOD:
                ts=ts[:T]  
            ts32=ts.astype(np.float32)
            np.save(d/f"{s}_ROITimeSeries{c}.npy",ts32)
            np.save(d/f"{s}_FCMatrixCondition{c}.npy",fc_of(ts32.astype(np.float64)))

def run_pipeline(seed=None):
    import importlib
    import dataset as ds_mod
    importlib.reload(ds_mod)
    from dataset import datasetPreparation
    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from attention_pool import condition_attention_pool
    from augmentations import graph_augmentor
    from train import joint_train
    from clustering import cluster as ClusterCls
    from torch.utils.data import DataLoader,random_split

    torch.manual_seed(config.randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.randomSeed)

    dataset=datasetPreparation(fm_only=False)

    class GW(torch.utils.data.Dataset):
        def __init__(self,sd): self.sd=sd
        def __len__(self): return len(self.sd)
        def __getitem__(self,i): return self.sd[i]

    grouped=GW(dataset.subjectData)
    nTot=len(grouped); nVal=int(nTot*config.valFraction)
    gen=torch.Generator().manual_seed(config.randomSeed)
    tr,va=random_split(grouped,[nTot-nVal,nVal],generator=gen)
    dataset.normalizeData(tr.indices)
    normStats=(dataset.nodeMean,dataset.nodeStd)

    trL=DataLoader(tr,batch_size=config.batchSize,shuffle=True,
                   collate_fn=lambda b:b,drop_last=True)
    vaL=DataLoader(va,batch_size=config.batchSize,shuffle=False,collate_fn=lambda b:b)

    enc=GNNEncoder().to(config.device)
    att=condition_attention_pool().to(config.device)
    enc,att,_,_=joint_train(enc,att,NTXentLoss(),trL,vaL,graph_augmentor(),
                             config.device,str(CKPT),normStats=normStats,
                             guardPrimary=True)

    cl=ClusterCls(enc,str(CKPT),config.conditions,dataset.subjectList)
    cl.deploy(dataset.subjectData)
    cl.setAttention(att)
    cl._split_fm_hc()
    best=cl.KMeansUse(skip_perm=True,skip_gap=True)
    tbl=best[0]
    kSel=int(tbl["k_selected_silhouette"].iloc[0])
    sil=float(tbl.loc[tbl["k"]==kSel,"silhouette_score"].iloc[0])
    nFM=len(cl.fmEmbed)
    del enc,att,dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sil,kSel,nFM,float(cl.hcSepSilh) if hasattr(cl,"hcSepSilh") else float("nan")

def main():
    lo=int(sys.argv[1]) if len(sys.argv)>1 else 0
    hi=int(sys.argv[2]) if len(sys.argv)>2 else 100
    print(f"propagated null: draws [{lo},{hi})  device={config.device}  "
          f"dModel={config.dModel} layers={config.layers}",flush=True)

    t0=time.perf_counter()
    model,lengths,F=fit_group_model()
    print(f"  group model fitted from {len(SUBS)} subjects x {len(CONDS)} conditions, "
          f"{F} frequency bins  ({time.perf_counter()-t0:.1f}s)",flush=True)

    type(config).subjectDataFolder=str(SURR)
    assert str(config.subjectDataFolder)==str(SURR)
    for i in range(lo,hi):
        out=RESULTS/f"draw{i:04d}.json"
        if out.exists():
            print(f"  draw {i:4d}  already done, skipping",flush=True); continue
        t=time.perf_counter()
        materialise(model,lengths,F,seed=900000+i)
        tgen=time.perf_counter()-t
        sil,k,nfm,hc=run_pipeline(seed=i)
        el=time.perf_counter()-t
        json.dump({"draw":i,"silhouette":sil,"k_selected":k,"n_fm":nfm,
                   "hc_sep_sil":hc,"gen_s":round(tgen,1),"total_s":round(el,1)},
                  open(out,"w"))
        print(f"  draw {i:4d}  sil={sil:.4f}  k={k}  nFM={nfm}  "
              f"[gen {tgen:.0f}s, total {el:.0f}s]",flush=True)

if __name__=="__main__":
    main()
