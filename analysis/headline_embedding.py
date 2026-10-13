#Headline cell recalibrated at 10,000 datasets
#  fitted to the real embedding
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
_R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_R/"experiments"/"ds002748_mdd"/"propagated_null"))
import nullcal as nc
import ov_common as ov
EMB=_R/"data"/"outputs"/"trained_fm_embeddings.npy";OUT=_R/"results"/"headline_embedding.json";N_DATASETS=10000;N_BLOCKS=5;N_DRAWS=1000;N_WORKERS=47;ROOT_SEED=20260917;EXPERIMENT_ID=33;_G={}
def _init(mean,factor,n):_G["mean"],_G["factor"],_G["n"]=(mean,factor,n)
def one(args):
    b,i,sd,d=args
    try:
        r=np.random.default_rng(sd);x=nc._draw(_G["mean"],_G["factor"],_G["n"],r);Xn=nc.l2_normalize(x);assert Xn.shape==(_G["n"],_G["factor"].shape[1]),Xn.shape;g=nc.min_cluster_guard(Xn.shape[0])
        s,k,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=g,n_init=ov.N_INIT,random_state=i)
        if k is None:return{"skip":True,"block":b,"i":i}
        kw=dict(k_grid=ov.K_GRID,n_draws=N_DRAWS,min_size=g,n_init=ov.N_INIT,seed=d,observed=(s,k));rM=nc.null_test(Xn,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,**kw);return dict(block=b,i=i,pM=float(rM.p),pC=float(rC.p),k=int(k),obs=float(s),pr=float(nc.participation_ratio(Xn)))
    except Exception:return{"error":traceback.format_exc(),"block":b,"i":i}
def rate(p,label):
    c=int((p<ov.ALPHA).sum())
    return dict(label=label,events=c,n=int(p.size),rejection_rate=c/p.size,wilson95=list(nc._wilson(c,p.size)),mean_p=float(p.mean()),median_p=float(np.median(p)),ks_D_vs_uniform=float(nc._ks_uniform(p)))
def main():
    import scipy
    import sklearn
    X=np.load(EMB).astype(np.float64);Xn=nc.l2_normalize(X);n,d=Xn.shape;m,f=nc._gaussian_factor(Xn);p=N_DATASETS//N_BLOCKS;a1=[]
    for b in range(N_BLOCKS):
        for i in range(p):a1.append((b,i,ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,0,b,i,0),ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,0,b,i,1)))
    assert len({a[2] for a in a1}|{a[3] for a in a1})==2*N_DATASETS;print("embedding %s, %d datasets x %d draws, %d blocks"%(X.shape,N_DATASETS,N_DRAWS,N_BLOCKS),flush=True);t0=time.time()
    with Pool(N_WORKERS,initializer=_init,initargs=(m,f,n))as pl:
        rs=[]
        for r in pl.imap_unordered(one,a1,chunksize=4):
            rs.append(r)
            if len(rs)%1000==0:
                el=time.time()-t0;print("  %d/%d  %.1f min elapsed, %.1f min remaining"%(len(rs),len(a1),el/60,el/60*(len(a1)-len(rs))/len(rs)),flush=True)
    e=[r for r in rs if "error" in r]
    if e:raise RuntimeError(e[0]["error"])
    r1=sorted((r for r in rs if "pM" in r),key=lambda r:(r["block"],r["i"]));pM=np.array([r["pM"] for r in r1]);pC=np.array([r["pC"] for r in r1]);bs=[]
    for b in range(N_BLOCKS):
        s=[r for r in r1 if r["block"]==b];a=np.array([r["pM"] for r in s])<ov.ALPHA;c=np.array([r["pC"] for r in s])<ov.ALPHA
        bs.append(dict(block=b,n_datasets=len(s),M_rate=float(a.mean()),C_rate=float(c.mean()),M_events=int(a.sum()),C_events=int(c.sum()),C_wilson95=list(nc._wilson(int(c.sum()),len(s)))))
    cr=[b["C_rate"] for b in bs]
    o=dict(experiment="headline cell recalibrated at the embedding geometry",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),embedding=str(EMB.relative_to(_R)).replace('\\',"/"),embedding_shape=[n,d],n_datasets=len(r1),n_draws=N_DRAWS,n_blocks=N_BLOCKS,n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,n_skipped=sum((1 for r in rs if r.get("skip"))),seed_rule="ov_common.hashed_seed(%d, %d, 0, block, i, purpose)"%(ROOT_SEED,EXPERIMENT_ID),differences_from_archived_run=["audited nullcal guard and search path","five k-means restarts rather than twenty"],pr=dict(mean=float(np.mean([r["pr"] for r in r1])),sd=float(np.std([r["pr"] for r in r1],ddof=1))),observed=dict(mean=float(np.mean([r["obs"] for r in r1]))),M=rate(pM,"M, unmatched"),C=rate(pC,"C, fully matched"),mcnemar=dict(M_only=int(((pM<ov.ALPHA)&(pC>=ov.ALPHA)).sum()),C_only=int(((pC<ov.ALPHA)&(pM>=ov.ALPHA)).sum())),blocks=bs,block_variance=dict(C_min=min(cr),C_max=max(cr),C_range_ratio=max(cr)/min(cr) if min(cr) else None),selected_k_counts={str(k):int(sum((1 for r in r1 if r["k"]==k))) for k in sorted({r["k"] for r in r1})},versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS},total_wall_s=round(time.time()-t0,1))
    OUT.write_text(json.dumps(o,indent=1))
    print("M %.4f %s   C %.4f %s   M-only %d  C-only %d  [%.0f min]"%(o["M"]["rejection_rate"],o["M"]["wilson95"],o["C"]["rejection_rate"],o["C"]["wilson95"],o["mcnemar"]["M_only"],o["mcnemar"]["C_only"],o["total_wall_s"]/60),flush=True)
    print("wrote %s"%OUT)
if __name__=="__main__":main()
