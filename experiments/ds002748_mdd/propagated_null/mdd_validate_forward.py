from __future__ import annotations
import os,sys,json,warnings,tempfile,shutil
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS","2")

RP=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",".."))
for sub in("","src","models","preprocessing",os.path.join("experiments","ds002748_mdd")):
    p=os.path.join(RP,sub) if sub else RP
    if p not in sys.path:
        sys.path.insert(0,p)

import numpy as np
import pandas as pd
import torch
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from config import config

_P=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"mdd_params.json")))
for _k,_v in _P.items():
    setattr(config,_k,_v)

import subject_filter_mdd as _sf
def _subs():
    return sorted(f.name for f in _sf.ROOT.iterdir()
                  if f.is_dir() and (_sf.ROOT/_sf.TS_PAT.format(s=f.name)).exists())
_sf.get_included_subjects_mdd=_subs

import dataset_mdd as dm

TR=dm.MDD_TR
BD=[(0.01,0.04),(0.04,0.10),(0.10,0.25)]


def fcr(ts):
    r=ConnectivityMeasure(kind="correlation",cov_estimator=LedoitWolf()).fit_transform([ts])[0]
    z=np.arctanh(r)
    np.fill_diagonal(z,0)
    z[np.isinf(z)]=0
    return z


def ndr(ts):
    d=pd.DataFrame(ts)
    m=d.mean(axis=0).to_numpy()
    v=d.var(ddof=1,axis=0).to_numpy()
    f=np.abs(np.fft.rfft(ts,axis=0,norm="forward"))**2
    l=np.fft.rfftfreq(n=ts.shape[0],d=TR)
    cs=[f[np.where((l>=lo)&(l<hi))[0],:].sum(axis=0) for lo,hi in BD]
    return np.column_stack([m,v]+cs)


def ndl(ts):
    d=pd.DataFrame(ts)
    out=[]
    for i in range(len(d.columns)):
        m=d.iloc[:,i].mean()
        v=d.iloc[:,i].var(ddof=1)
        x=d.iloc[:,i].to_numpy()
        f=np.abs(np.fft.rfft(x,axis=0,norm="forward"))**2
        l=np.fft.rfftfreq(n=len(x),d=TR)
        ix=[np.where((l>=lo)&(l<hi)) for lo,hi in BD]
        out.append([m,v]+[float(np.sum(f[j[0]])) for j in ix])
    return np.array(out)


def edr(z):
    w=np.where(np.abs(z)>=np.percentile(np.abs(z),config.edgePercentile))
    return np.array([w[0],w[1]]),z[w[0],w[1]]


def main():
    ds=dm.mddDataset()
    print(f"dataset_mdd {len(ds.DataList)} subjects x {tuple(ds.DataList[0].x.shape)} seed {config.randomSeed}",flush=True)
    assert ds.rawX is not None,"expected dataset_mdd.normalize() to have cached rawX"

    we=wi=wa=0.0
    wx=0
    rw=[]
    for i,g in enumerate(ds.DataList):
        s=g.subjectID
        ts=np.load(dm.MDD_ROOT/f"{s}/{s}_rest_ts.npy")
        ng=ds.rawX[i].numpy()
        de=float(np.max(np.abs(ndl(ts)-ng)))
        di=float(np.max(np.abs(ndr(ts.astype(np.float64))-ndl(ts.astype(np.float64)))))
        zr=fcr(ts)
        xr,ar=edr(zr)
        dx=int(np.max(np.abs(xr-g.edge_index.numpy()))) if xr.shape==g.edge_index.numpy().shape else -1
        da=float(np.max(np.abs(ar-g.edge_attr.numpy())))
        we=max(we,de)
        wi=max(wi,di)
        wa=max(wa,da)
        wx=max(wx,dx)
        rw.append((s,de,di,da,dx,xr.shape[1]))

    print(f"  over {len(rw)} subjects",flush=True)
    print(f"    nodes pipeline vs stored   {we:.3e}",flush=True)
    print(f"    nodes independent float64  {wi:.3e}",flush=True)
    print(f"    edge weights vs indep      {wa:.3e}",flush=True)
    print(f"    edge index vs indep        {wx:d}",flush=True)

    tp=tempfile.mkdtemp(prefix="mdd_fwd_")
    try:
        ot=oc=on=0
        for g in ds.DataList:
            s=g.subjectID
            ts=np.load(dm.MDD_ROOT/f"{s}/{s}_rest_ts.npy")
            os.makedirs(os.path.join(tp,s),exist_ok=True)
            q=os.path.join(tp,s,f"{s}_rest_ts.npy")
            np.save(q,ts.astype(np.float64).astype(np.float32))
            rt=np.load(q)
            ot+=int(np.array_equal(ts,rt))
            oc+=int(np.array_equal(fcr(rt),fcr(ts)))
            on+=int(np.array_equal(ndr(rt),ndr(ts)))
        n=len(ds.DataList)
        print(f"  round trip {n} subjects: ts {ot}/{n} fc {oc}/{n} nodes {on}/{n}",flush=True)
    finally:
        shutil.rmtree(tp,ignore_errors=True)

    vd=(we==0.0) and (wi<1e-9) and (wa<1e-6) and (wx==0) and ot==oc==on==n
    print(f"  forward path {'REPRODUCES' if vd else 'DOES NOT MATCH'}",flush=True)

    json.dump({"n_subjects":len(rw),
               "node_pipeline_vs_stored":we,
               "node_independent_impl_float64":wi,
               "worst_edge_weight_deviation":wa,
               "worst_edge_index_deviation":wx,
               "roundtrip_ts_identical":ot,
               "roundtrip_fc_identical":oc,
               "roundtrip_nodes_identical":on,
               "verdict":"reproduces" if vd else "does not match"},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"mdd_forward_validation.json"),"w"),indent=2)


if __name__=="__main__":
    main()
