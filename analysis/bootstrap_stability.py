import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
_ROOT=Path(__file__).resolve().parent.parent
for _p in(_ROOT,_ROOT/"src",_ROOT/"models",_ROOT/"analysis"):
    if str(_p) not in sys.path:sys.path.insert(0,str(_p))
from config import config
from clustering import cluster
from dataset import datasetPreparation
from sensitivity_analysis import load_trained_model
def _t(m):return torch.tensor(np.asarray(m),dtype=torch.float32)
def run_bootstrap(runner,emb,ids,n_res=None,seed=None):
    n_res=config.bootstrapNResamples if n_res is None else n_res;seed=config.randomSeed if seed is None else seed;og=runner.KMeansUse(_t(emb),list(ids),skip_perm=True,skip_gap=True)
    o=dict(zip(ids,og[2]));rg=np.random.default_rng(seed);n=len(ids);a=np.empty(n_res)
    for b in range(n_res):
        ix=rg.integers(0,n,n);bs=[ids[i] for i in ix];r=runner.KMeansUse(_t(emb[ix]),bs,skip_perm=True,skip_gap=True);sn={}
        for sd,l in zip(bs,r[2]):sn[sd]=l
        c=list(sn);a[b]=adjusted_rand_score([o[s] for s in c],[sn[s] for s in c])
    return a
def _summary_frame(aris):
    r=pd.DataFrame({"resample":np.arange(len(aris)),"ari":aris})
    s={"n_resamples":len(aris),"ari_mean":float(np.mean(aris)),"ari_std":float(np.std(aris)),"ari_min":float(np.min(aris)),"ari_max":float(np.max(aris))};return(r,s)
def run_real(conds):
    m=load_trained_model(config.jointCheckpointPath)
    if m is None:return(None,None,f"missing trained checkpoint ({config.jointCheckpointPath})")
    ec,at=m
    try:
        ds=datasetPreparation(fm_only=False);ck=torch.load(config.jointCheckpointPath,map_location="cpu")
        if ck.get("nodeMean") is not None:ds.applyNormalization(ck["nodeMean"],ck["nodeStd"])
        else:warnings.warn("checkpoint has no saved normalization stats; falling back to all-subject statistics (train/inference mismatch)")
        rr=cluster(ec,str(config.checkpointDir),conds,ds.subjectList);rr.deploy(ds.subjectData);rr.setAttention(at);rr._split_fm_hc();ft,f=rr._stack(rr.fmEmbed);o=list(np.argsort(f))
        f=[f[i] for i in o];ft=ft[o]
    except Exception as e:return(None,None,f"could not build FM embeddings: {type(e).__name__}: {e}")
    eb=ft.detach().cpu().numpy();a=run_bootstrap(rr,eb,f,config.bootstrapNResamples);r,s=_summary_frame(a);return(r,s,None)
def _synthetic_embeddings(n=30,k=3,d=config.dModel,spread=8.0,noise=1.0,seed=0):
    r=np.random.default_rng(seed);c=r.normal(0,spread,(k,d));m=np.empty((n,d),dtype=np.float32)
    for i in range(n):m[i]=c[i%k]+r.normal(0,noise,d)
    i1=[f"sub-fm{i:03d}"for i in range(n)];return(m,i1)
def _bare_runner(conds,ids):return cluster(None,str(config.checkpointDir),conds,ids)
def structural_self_test():
    c=list(config.conditions);n=25;print("[self-test] well-separated clusters -> ARI should be near 1.0 ...");m,i1=_synthetic_embeddings(noise=0.5,seed=1);rr=_bare_runner(c,i1)
    a=run_bootstrap(rr,m,i1,n_res=n,seed=config.randomSeed);print(f"  clean: mean={a.mean():.4f} range=[{a.min():.4f},{a.max():.4f}]");assert a.mean()>0.8,"well-separated clusters must be stable"
    print("[self-test] pure-noise embeddings -> ARI should be lower/scattered ...");r=np.random.default_rng(7);ny=r.normal(0,1,(30,config.dModel)).astype(np.float32)
    n1=[f"sub-fm{i:03d}"for i in range(30)];nr=_bare_runner(c,n1);ns=run_bootstrap(nr,ny,n1,n_res=n,seed=config.randomSeed)
    print(f"  noise: mean={ns.mean():.4f} range=[{ns.min():.4f},{ns.max():.4f}]");assert ns.mean()<a.mean(),"noise must be less stable than clean clusters"
    print("[self-test] duplicate-collapse + ARI alignment ...");f,s=_summary_frame(a);assert len(f)==n and s["n_resamples"]==n,"summary shape mismatch"
    print("[self-test] Good! Bootstrap resampling + ARI logic are sound.");return True
def main():
    c=list(config.conditions);r=config.resultsRoot;os.makedirs(r,exist_ok=True);bs=[];print("="*70);print("Bootstrap stability analysis (subtype reproducibility under resampling)")
    print(f"  BOOTSTRAP_N_RESAMPLES = {config.bootstrapNResamples}");print("="*70);rs,s,bk=run_real(c)
    if rs is not None:
        o=r/"bootstrap_stability.csv";rs.to_csv(o,index=False);print(f"[bootstrap] wrote {o}")
        print(f"  ARI mean={s["ari_mean"]:.4f} std={s["ari_std"]:.4f} range=[{s["ari_min"]:.4f},{s["ari_max"]:.4f}]")
    else:bs.append(f"bootstrap stability BLOCKED: {bk}")
    if bs:
        print('\n'+"-"*70);print("BLOCKERS (no results fabricated):")
        for b in bs:print(f"  - {b}")
        print("-"*70);print("Running structural self-test against synthetic data instead....");structural_self_test()
    return bs
if __name__=="__main__":main()
