#Is the reversal an artifact of the k-means restart count?
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
SHAPE=_R/"results"/"overnight"/"e2_spectrum_shape.json";OUT=_R/"results"/"restart_sensitivity.json";N=28;NDS=400;N_DRAWS=200;RESTARTS=[5,20];CELLS=["two_block_pr4.5","exp_pr4.5"];N_WORKERS=47
ROOT_SEED=20260918;EXPERIMENT_ID=37;_LAM=None
def _init(lam):
    global _LAM;_LAM=lam
def one(args):
    i,sd,d,n=args
    try:
        Xn=ov.gen(N,ov.D,_LAM,sd);g=nc.min_cluster_guard(N);s,k,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=g,n_init=n,random_state=i)
        if k is None:return{"skip":True,"i":i}
        kw=dict(k_grid=ov.K_GRID,n_draws=N_DRAWS,min_size=g,n_init=n,seed=d,observed=(s,k))
        return dict(i=i,k=int(k),pM=float(nc.null_test(Xn,reproject=False,match_selection=False,**kw).p),pC=float(nc.null_test(Xn,reproject=True,match_selection=True,**kw).p))
    except Exception:return{"error":traceback.format_exc(),"i":i}
def rate(c,n):return dict(events=int(c),n=int(n),rate=c/n,wilson95=list(nc._wilson(int(c),int(n))))
def spectrum(cell):
    f,pv=(cell["family"],cell["param_value"])
    if f=="exp":return ov.spec_exp(ov.D,float(pv))
    if f=="two_block":
        m,e=str(pv).split("/");return ov.spec_two_block(ov.D,int(m),float(e))
    raise ValueError(f)
def main():
    import scipy
    import sklearn
    s=json.loads(SHAPE.read_text())["cells"]
    o=dict(experiment="restart sensitivity of the reversal",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),n=N,d=ov.D,n_datasets=NDS,n_draws=N_DRAWS,restarts=RESTARTS,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,seed_rule="ov_common.hashed_seed(%d, %d, cell, restart index, dataset, purpose)"%(ROOT_SEED,EXPERIMENT_ID),versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS},cells={})
    t0=time.time()
    for c,k in enumerate(CELLS):
        l=spectrum(s[k]);o["cells"][k]=dict(family=s[k]["family"],param_value=s[k]["param_value"],archived=dict(rej_M=s[k]["rej_M"],rej_C=s[k]["rej_C"]),by_restarts={})
        for rx,n in enumerate(RESTARTS):
            tc=time.time();a=[(i,ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,c,rx,i,0),ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,c,rx,i,1),n) for i in range(NDS)]
            with Pool(N_WORKERS,initializer=_init,initargs=(l,))as p:rs=list(p.imap_unordered(one,a,chunksize=1))
            e=[r for r in rs if "error" in r]
            if e:raise RuntimeError(e[0]["error"])
            r1=[r for r in rs if "pM" in r];pM=np.array([r["pM"] for r in r1]);pC=np.array([r["pC"] for r in r1])
            o["cells"][k]["by_restarts"][str(n)]=dict(n_init=n,n_datasets=len(r1),M=rate(int((pM<ov.ALPHA).sum()),len(r1)),C=rate(int((pC<ov.ALPHA).sum()),len(r1)),C_only=int(((pC<ov.ALPHA)&(pM>=ov.ALPHA)).sum()),M_only=int(((pM<ov.ALPHA)&(pC>=ov.ALPHA)).sum()),wall_s=round(time.time()-tc,1))
            r_=o["cells"][k]["by_restarts"][str(n)]
            print("[%s] restarts %2d  M %.4f  C %.4f  C-only %d  M-only %d  [%.0fs]"%(k,n,r_["M"]["rate"],r_["C"]["rate"],r_["C_only"],r_["M_only"],r_["wall_s"]),flush=True)
            OUT.write_text(json.dumps(o,indent=1))
    o["total_wall_s"]=round(time.time()-t0,1);OUT.write_text(json.dumps(o,indent=1));print("wrote %s"%OUT)
if __name__=="__main__":main()
