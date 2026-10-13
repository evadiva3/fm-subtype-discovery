#The two errors separated at the headline geometry, on one set of datasets.
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
EMB=_R/"data"/"outputs"/"trained_fm_embeddings.npy";OUT=_R/"results"/"two_error_decomposition.json";N_DATASETS=2000;N_DRAWS=1000;N_WORKERS=47;ROOT_SEED=20260918;EXPERIMENT_ID=36
ARMS={"M":dict(reproject=False,match_selection=False),"G":dict(reproject=True,match_selection=False),"S":dict(reproject=False,match_selection=True),"C":dict(reproject=True,match_selection=True)};_G={}
def _init(mean,factor,n):_G["mean"],_G["factor"],_G["n"]=(mean,factor,n)
def one(args):
    i,sd,d=args
    try:
        r=np.random.default_rng(sd);Xn=nc.l2_normalize(nc._draw(_G["mean"],_G["factor"],_G["n"],r));g=nc.min_cluster_guard(Xn.shape[0])
        s,k,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=g,n_init=ov.N_INIT,random_state=i)
        if k is None:return{"skip":True,"i":i}
        kw=dict(k_grid=ov.K_GRID,n_draws=N_DRAWS,min_size=g,n_init=ov.N_INIT,seed=d,observed=(s,k));o=dict(i=i,k=int(k),obs=float(s),pr=float(nc.participation_ratio(Xn)))
        for am,a in ARMS.items():o["p_"+am]=float(nc.null_test(Xn,**a,**kw).p)
        return o
    except Exception:return{"error":traceback.format_exc(),"i":i}
def rate(p,label):
    c=int((p<ov.ALPHA).sum())
    return dict(label=label,events=c,n=int(p.size),rejection_rate=c/p.size,wilson95=list(nc._wilson(c,p.size)),mean_p=float(p.mean()),ks_D_vs_uniform=float(nc._ks_uniform(p)))
def main():
    import scipy
    import sklearn
    X=np.load(EMB).astype(np.float64);Xn=nc.l2_normalize(X);n,d=Xn.shape;m,f=nc._gaussian_factor(Xn)
    a1=[(i,ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,0,0,i,0),ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,0,0,i,1)) for i in range(N_DATASETS)];assert len({a[1] for a in a1}|{a[2] for a in a1})==2*N_DATASETS
    print("%d datasets x %d draws, four arms, embedding %s"%(N_DATASETS,N_DRAWS,X.shape),flush=True);t0=time.time()
    with Pool(N_WORKERS,initializer=_init,initargs=(m,f,n))as pl:
        rs=[]
        for r in pl.imap_unordered(one,a1,chunksize=4):
            rs.append(r)
            if len(rs)%500==0:
                el=time.time()-t0;print("  %d/%d  %.1f min elapsed"%(len(rs),len(a1),el/60),flush=True)
    e=[r for r in rs if "error" in r]
    if e:raise RuntimeError(e[0]["error"])
    r1=sorted((r for r in rs if "p_M" in r),key=lambda r:r["i"]);p={a:np.array([r["p_"+a] for r in r1]) for a in ARMS};ps={}
    for a,b in(("M","G"),("M","S"),("M","C"),("G","C"),("S","C")):
        ra,rb=(p[a]<ov.ALPHA,p[b]<ov.ALPHA);ps["%s_vs_%s"%(a,b)]=dict(a_only=int((ra&~rb).sum()),b_only=int((rb&~ra).sum()))
    o=dict(experiment="geometry-only and selection-only arms at the headline geometry",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),embedding=str(EMB.relative_to(_R)).replace('\\',"/"),embedding_shape=[n,d],n_datasets=len(r1),n_draws=N_DRAWS,n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,n_skipped=sum((1 for r in rs if r.get("skip"))),seed_rule="ov_common.hashed_seed(%d, %d, 0, 0, i, purpose)"%(ROOT_SEED,EXPERIMENT_ID),arm_definitions={"M":"neither error repaired","G":"geometry matched only","S":"selection matched only","C":"both matched"},pr=dict(mean=float(np.mean([r["pr"] for r in r1])),sd=float(np.std([r["pr"] for r in r1],ddof=1))),arms={a:rate(p[a],a) for a in ARMS},paired=ps,versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS},total_wall_s=round(time.time()-t0,1))
    OUT.write_text(json.dumps(o,indent=1));print("  ".join(("%s %.4f %s"%(a,o["arms"][a]["rejection_rate"],o["arms"][a]["wilson95"]) for a in ARMS)),flush=True);print("wrote %s"%OUT)
if __name__=="__main__":main()
