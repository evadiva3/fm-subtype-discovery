#Reversal verification on seed blocks that share no datasets.
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
D=ov.D;N=28;NDS=400;N_DRAWS=200;DEEP_DRAWS=2000;N_RUNS=4;N_WORKERS=47;ROOT_SEED=20260917;EXPERIMENT_ID=32;SHAPE=_R/"results"/"overnight"/"e2_spectrum_shape.json"
OUT=_R/"results"/"reversal_verification.json"
CELLS=["exp_pr4.5","power_pr3.0","power_pr4.5","power_pr6.0","one_spike_pr3.0","one_spike_pr4.5","one_spike_pr6.0","two_block_pr3.0","two_block_pr4.5","two_block_pr6.0"];_LAM=None
def spectrum(cell):
    f=cell["family"];pv=cell["param_value"]
    if f=="exp":return ov.spec_exp(D,float(pv))
    if f=="power":return ov.spec_power(D,float(pv))
    if f=="one_spike":return ov.spec_one_spike(D,float(pv))
    if f=="two_block":
        m,e=str(pv).split("/");return ov.spec_two_block(D,int(m),float(e))
    raise ValueError(f)
def _init(lam):
    global _LAM;_LAM=lam
def one(args):
    r,i,sd,d,n=args
    try:
        Xn=ov.gen(N,D,_LAM,sd);g=nc.min_cluster_guard(N);s,k,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=g,n_init=ov.N_INIT,random_state=i)
        if k is None:return{"skip":True,"run":r,"i":i}
        kw=dict(k_grid=ov.K_GRID,n_draws=n,min_size=g,n_init=ov.N_INIT,seed=d,observed=(s,k));rM=nc.null_test(Xn,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,**kw)
        return dict(run=r,i=i,seed=str(sd),draw_seed=str(d),pM=float(rM.p),pC=float(rC.p),obs=float(s),k=int(k),pr=float(nc.participation_ratio(Xn)))
    except Exception:return{"error":traceback.format_exc(),"run":r,"i":i}
def seeds(cell_id,run,i):return(ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,cell_id,run,i,0),ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,cell_id,run,i,1))
def rate(c,n):return dict(events=int(c),n=int(n),rate=c/n,wilson95=list(nc._wilson(int(c),int(n))))
def main():
    import scipy
    import sklearn
    se=json.loads(SHAPE.read_text())["cells"];s1=json.loads(OUT.read_text()) if OUT.exists() else{"cells":{}}
    s1["config"]=dict(experiment="reversal verification on disjoint seed blocks",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),n=N,d=D,n_datasets=NDS,n_draws=N_DRAWS,deep_draws=DEEP_DRAWS,n_runs=N_RUNS,n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,seed_rule="ov_common.hashed_seed(%d, %d, cell_id, run, i, purpose); purpose 0 data, 1 draws; observed search random_state=i"%(ROOT_SEED,EXPERIMENT_ID),spectra_from="results/overnight/e2_spectrum_shape.json",versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS})
    t0=time.time()
    for cd,k in enumerate(CELLS):
        if k in s1["cells"]:
            print("skip %s"%k,flush=True);continue
        cl=se[k];l=spectrum(cl);tc=time.time();a1=[]
        for rn in range(N_RUNS):
            for i in range(NDS):
                ds,dw=seeds(cd,rn,i);a1.append((rn,i,ds,dw,N_DRAWS))
        assert len({a[2] for a in a1}|{a[3] for a in a1})==2*N_RUNS*NDS
        with Pool(N_WORKERS,initializer=_init,initargs=(l,))as pl:
            rs=[r for r in pl.imap_unordered(one,a1,chunksize=1)];e=[r for r in rs if "error" in r]
            if e:raise RuntimeError(e[0]["error"])
            r1=sorted((r for r in rs if "pM" in r),key=lambda r:(r["run"],r["i"]));dc=[r for r in r1 if(r["pM"]<ov.ALPHA)!=(r["pC"]<ov.ALPHA)]
            d=list(pl.imap_unordered(one,[(r["run"],r["i"],int(r["seed"]),int(r["draw_seed"]),DEEP_DRAWS) for r in dc],chunksize=1)) if dc else[]
        by={(r["run"],r["i"]):r for r in d if "pM" in r};p=0
        for r in dc:
            g=by.get((r["run"],r["i"]));w=r["pC"]<ov.ALPHA
            if g is None:
                r["deep"]=None;continue
            n=(g["pM"]<ov.ALPHA)!=(g["pC"]<ov.ALPHA);s=bool(n and(g["pC"]<ov.ALPHA)==w);r["deep"]=dict(pM=g["pM"],pC=g["pC"],persists=s);p+=int(s and w)
        pM=np.array([r["pM"] for r in r1]);pC=np.array([r["pC"] for r in r1]);c=int(((pC<ov.ALPHA)&(pM>=ov.ALPHA)).sum());m=int(((pM<ov.ALPHA)&(pC>=ov.ALPHA)).sum());r2=[]
        for rn in range(N_RUNS):
            sb=[r for r in r1 if r["run"]==rn];a=np.array([r["pM"] for r in sb])<ov.ALPHA;b=np.array([r["pC"] for r in sb])<ov.ALPHA
            r2.append(dict(run=rn,n_datasets=len(sb),M_events=int(a.sum()),C_events=int(b.sum()),C_only=int((b&~a).sum()),M_only=int((a&~b).sum())))
        o=dict(family=cl["family"],param_name=cl["param_name"],param_value=cl["param_value"],pr_mean=float(np.mean([r["pr"] for r in r1])),archived=dict(rej_M=cl["rej_M"],rej_C=cl["rej_C"]),n_datasets=len(r1),M=rate(int((pM<ov.ALPHA).sum()),len(r1)),C=rate(int((pC<ov.ALPHA).sum()),len(r1)),C_only=c,M_only=m,runs=r2,deep=dict(draws=DEEP_DRAWS,tested=len(dc),C_only_persisting=p,persisting_fraction=p/c if c else None),discordant=dc,wall_s=round(time.time()-tc,1))
        s1["cells"][k]=o;OUT.write_text(json.dumps(s1,indent=1))
        print("[%s] PR %.2f  M %.4f  C %.4f  C-only %d  M-only %d  persist %d/%d  [%.0fs]"%(k,o["pr_mean"],o["M"]["rate"],o["C"]["rate"],c,m,p,c,o["wall_s"]),flush=True)
    s1["config"]["total_wall_s"]=round(time.time()-t0,1);OUT.write_text(json.dumps(s1,indent=1));print("wrote %s"%OUT)
if __name__=="__main__":main()
