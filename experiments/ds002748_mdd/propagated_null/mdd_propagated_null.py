from __future__ import annotations
import os,sys,json,time,warnings,gc,io,contextlib
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS","2")

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

ROOT=Path(os.environ.get("MDD_NULL_ROOT",os.path.join(RP,"results","mdd","propagated_null","_run")))
WORK=ROOT/"work"
SURR=WORK/"Subjects"
CK=WORK/"ckpt"
RES=ROOT/"draws"
for _d in(SURR,CK,RES):
    _d.mkdir(parents=True,exist_ok=True)

REAL=_dm.MDD_ROOT
TM=100
TSN="{s}/{s}_rest_ts.npy"


SUBS=_subs()


def fit():
    X,LN=[],{}
    for s in SUBS:
        ts=np.load(REAL/TSN.format(s=s)).astype(np.float64)
        LN[s]=ts.shape[0]
        if ts.shape[0]==TM:
            X.append(ts)
    X=np.asarray(X)
    assert X.shape[0]==len(SUBS)
    Xf=np.fft.rfft(X,axis=1)
    n,F,P=Xf.shape
    As=[]
    for f in range(F):
        z=Xf[:,f,:]
        S=(z.conj().T@z)/n
        w,V=np.linalg.eigh(S)
        kp=w>max(w.max(),0)*1e-12
        As.append((V[:,kp]*np.sqrt(w[kp])).astype(np.complex128))
    return As,LN,F


def dsub(As,F,rng):
    P=As[0].shape[0]
    Z=np.zeros((F,P),dtype=np.complex128)
    for f in range(F):
        A=As[f]
        r=A.shape[1]
        if r==0:
            continue
        if f==0 or f==F-1:
            Z[f]=A@rng.standard_normal(r)
        else:
            w=(rng.standard_normal(r)+1j*rng.standard_normal(r))/np.sqrt(2.0)
            Z[f]=A@w
    return np.fft.irfft(Z,n=TM,axis=0)


def matr(As,LN,F,seed):
    rng=np.random.default_rng(seed)
    for s in SUBS:
        d=SURR/s
        d.mkdir(parents=True,exist_ok=True)
        ts=dsub(As,F,rng)
        T=LN.get(s,TM)
        if T!=TM:
            ts=ts[:T]
        np.save(d/f"{s}_rest_ts.npy",ts.astype(np.float32))


def run(root=None):
    import importlib
    import dataset_mdd as dsm
    importlib.reload(dsm)
    dsm.MDD_ROOT=Path(root) if root is not None else SURR

    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from augmentations import graph_augmentor
    from train_mdd import train_mdd
    from clustering_mdd import mddCluster
    from torch.utils.data import DataLoader,random_split

    torch.manual_seed(config.randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.randomSeed)

    ds=dsm.mddDataset()

    class W(torch.utils.data.Dataset):
        def __init__(self,sd):self.sd=sd
        def __len__(self):return len(self.sd)
        def __getitem__(self,i):return self.sd[i]

    gd=W(ds.subjectData)
    n=len(gd)
    nv=int(n*config.valFraction)
    nt=n-nv
    gen=torch.Generator().manual_seed(config.randomSeed)
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
    tbl=cl.KMeansUse(t,ids,skip_perm=True,skip_gap=True)[0]
    k=int(tbl["k_selected_silhouette"].iloc[0])
    sil=float(tbl.loc[tbl["k"]==k,"silhouette_score"].iloc[0])

    aid=list(cl.attentionEmbeddings.keys())
    E=torch.stack([cl.attentionEmbeddings[i] for i in aid]).detach().cpu().numpy()
    gl=np.array([cl.groupLabels[i] for i in aid])
    E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
    sep=float(silhouette_score(E,gl))

    hcT,_=cl._stack(cl.hcEmbed)
    prj=cl.project_ortho(t,hcT)
    oT=cl.KMeansUse(prj,ids,skip_perm=True,skip_gap=True)[0]
    oK=int(oT["k_selected_silhouette"].iloc[0])
    oS=float(oT.loc[oT["k"]==oK,"silhouette_score"].iloc[0])

    nM=len(cl.fmEmbed)
    nH=len(cl.hcEmbed)
    del enc,ds,cl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(silhouette=sil,k_selected=k,n_mdd=nM,n_hc=nH,
                sep_sil=sep,ortho_silhouette=oS,ortho_k=oK)


def main():
    md=sys.argv[1] if len(sys.argv)>1 else "0"

    if md=="observed":
        print(f"observed device={config.device} dModel={config.dModel} heads={config.heads} layers={config.layers}",flush=True)
        t=time.perf_counter()
        r=run(REAL)
        r["elapsed_s"]=round(time.perf_counter()-t,1)
        json.dump(r,open(ROOT/"observed_thismachine.json","w"),indent=2)
        print(f"  k={r['k_selected']} sil={r['silhouette']:.4f} sep={r['sep_sil']:+.4f} [{r['elapsed_s']:.0f}s]",flush=True)
        return

    lo=int(sys.argv[1]) if len(sys.argv)>1 else 0
    hi=int(sys.argv[2]) if len(sys.argv)>2 else 200
    print(f"draws [{lo},{hi}) device={config.device} dModel={config.dModel} heads={config.heads} layers={config.layers}",flush=True)

    As,LN,F=fit()
    assert SURR!=REAL
    for i in range(lo,hi):
        q=RES/f"draw{i:04d}.json"
        if q.exists():
            continue
        t=time.perf_counter()
        matr(As,LN,F,seed=900000+i)
        tg=time.perf_counter()-t
        r=run()
        el=time.perf_counter()-t
        r.update(draw=i,gen_s=round(tg,1),total_s=round(el,1))
        json.dump(r,open(q,"w"))
        print(f"  {i:4d} k={r['k_selected']} sil={r['silhouette']:.4f} sep={r['sep_sil']:+.4f} ortho={r['ortho_silhouette']:.4f} [gen {tg:.0f}s, {el:.0f}s]",flush=True)


if __name__=="__main__":
    main()
