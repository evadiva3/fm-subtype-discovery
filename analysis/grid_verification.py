#Grid-cell verification on seed blocks that share no datasets.
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
ARCHIVE=_R/"results"/"overnight"/"e1_conly_check.json";OUT=_R/"results"/"grid_verification.json";N_RUNS=4;N_WORKERS=47;ROOT_SEED=20260918;EXPERIMENT_ID=35;_LAM=None
def _init(lam):
    global _LAM;_LAM=lam
def one(args):
    r,i,sd,d,ns,n=args
    try:
        Xn=ov.gen(n,ov.D,_LAM,sd);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=g,n_init=ov.N_INIT,random_state=i)
        if k is None:return{"skip":True,"run":r,"i":i}
        kw=dict(k_grid=ov.K_GRID,n_draws=ns,min_size=g,n_init=ov.N_INIT,seed=d,observed=(s,k));rM=nc.null_test(Xn,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,**kw)
        return dict(run=r,i=i,seed=str(sd),draw_seed=str(d),pM=float(rM.p),pC=float(rC.p),obs=float(s),k=int(k),pr=float(nc.participation_ratio(Xn)))
    except Exception:return{"error":traceback.format_exc(),"run":r,"i":i}
def _rel(p):
    try:return str(Path(p).relative_to(_R)).replace('\\',"/")
    except ValueError:return str(p)
def rate(c,n):return dict(events=int(c),n=int(n),rate=c/n,wilson95=list(nc._wilson(int(c),int(n))))
def main():
    import scipy
    import sklearn
    ah=json.loads(ARCHIVE.read_text());cg=ah["config"];n1,ns,d=(cg["n_datasets"],cg["n_draws"],cg["deep_draws"]);ks=sys.argv[1:] or list(ah["cells"])
    s1=json.loads(OUT.read_text()) if OUT.exists() else{"cells":{}}
    s1["config"]=dict(experiment="grid-cell verification on disjoint seed blocks",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),d=ov.D,n_datasets=n1,n_draws=ns,deep_draws=d,n_runs=N_RUNS,n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,archived=_rel(ARCHIVE),seed_rule="ov_common.hashed_seed(%d, %d, cell index, run, dataset, purpose)"%(ROOT_SEED,EXPERIMENT_ID),versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS})
    t0=time.time()
    for cd,k in enumerate(ah["cells"]):
        if k not in ks or k in s1["cells"]:continue
        cl=ah["cells"][k];n=cl["n"];l=ov.spec_exp(ov.D,cl["tau"]);tc=time.time();a1=[]
        for rn in range(N_RUNS):
            for i in range(n1):a1.append((rn,i,ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,cd,rn,i,0),ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,cd,rn,i,1),ns,n))
        assert len({a[2] for a in a1}|{a[3] for a in a1})==2*N_RUNS*n1
        with Pool(N_WORKERS,initializer=_init,initargs=(l,))as pl:
            rs=list(pl.imap_unordered(one,a1,chunksize=1));e=[r for r in rs if "error" in r]
            if e:raise RuntimeError(e[0]["error"])
            r1=sorted((r for r in rs if "pM" in r),key=lambda r:(r["run"],r["i"]));dc=[r for r in r1 if(r["pM"]<ov.ALPHA)!=(r["pC"]<ov.ALPHA)]
            ds=list(pl.imap_unordered(one,[(r["run"],r["i"],int(r["seed"]),int(r["draw_seed"]),d,n) for r in dc],chunksize=1)) if dc else[]
        by={(r["run"],r["i"]):r for r in ds if "pM" in r};p=0
        for r in dc:
            g=by.get((r["run"],r["i"]));w=r["pC"]<ov.ALPHA
            if g is None:
                r["deep"]=None;continue
            nw=(g["pM"]<ov.ALPHA)!=(g["pC"]<ov.ALPHA);se=bool(nw and(g["pC"]<ov.ALPHA)==w);r["deep"]=dict(pM=g["pM"],pC=g["pC"],persists=se);p+=int(se and w)
        pM=np.array([r["pM"] for r in r1]);pC=np.array([r["pC"] for r in r1]);c=int(((pC<ov.ALPHA)&(pM>=ov.ALPHA)).sum());m=int(((pM<ov.ALPHA)&(pC>=ov.ALPHA)).sum())
        s1["cells"][k]=dict(cell=k,n=n,tau=cl["tau"],pr_target=cl["pr_target"],pr_mean=float(np.mean([r["pr"] for r in r1])),n_datasets=len(r1),M=rate(int((pM<ov.ALPHA).sum()),len(r1)),C=rate(int((pC<ov.ALPHA).sum()),len(r1)),C_only=c,M_only=m,runs=[dict(run=r,M_events=int(sum((1 for x in r1 if x["run"]==r and x["pM"]<ov.ALPHA))),C_events=int(sum((1 for x in r1 if x["run"]==r and x["pC"]<ov.ALPHA)))) for r in range(N_RUNS)],deep=dict(draws=d,tested=len(dc),C_only_persisting=p,persisting_fraction=p/c if c else None),archived=dict(total_C_only=cl["total_C_only"],total_datasets=cl["total_datasets"],deep=cl["deep"]),discordant=dc,wall_s=round(time.time()-tc,1))
        OUT.write_text(json.dumps(s1,indent=1));s=s1["cells"][k]
        print("[%s] n=%d PR %.2f  M %.4f  C %.4f  C-only %d  M-only %d  persist %d/%d  [%.0fs]"%(k,n,s["pr_mean"],s["M"]["rate"],s["C"]["rate"],c,m,p,c,s["wall_s"]),flush=True)
    s1["config"]["total_wall_s"]=round(time.time()-t0,1);OUT.write_text(json.dumps(s1,indent=1));print("wrote %s"%OUT)
if __name__=="__main__":main()
