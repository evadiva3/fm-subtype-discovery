#Within/between and Davies-Bouldin verification on seed blocks that share no datasets.
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
from e6_statistics import best_stat,stat_at_k
ARCHIVE=_R/"results"/"overnight"/"e16_nwb_check.json";OUT=_R/"results"/"statistic_verification.json";N_RUNS=4;N_WORKERS=47;ROOT_SEED=20260918;EXPERIMENT_ID=34;_LAM=None
def _init(lam):
    global _LAM;_LAM=lam
def one(args):
    rn,i,s,d,ns,st,n=args
    try:
        Xn=ov.gen(n,ov.D,_LAM,s);g=nc.min_cluster_guard(n);o,k=best_stat(st,Xn,ov.K_GRID,g,ov.N_INIT,i)
        if k is None:return{"skip":True,"run":rn,"i":i}
        rm=np.random.default_rng(d);r=np.random.default_rng(d);m,A=nc._gaussian_factor(Xn);sM=np.empty(ns);sC=np.empty(ns)
        for b in range(ns):
            sM[b]=stat_at_k(st,nc._draw(m,A,n,rm),k,ov.N_INIT,b);sC[b]=best_stat(st,nc.l2_normalize(nc._draw(m,A,n,r)),ov.K_GRID,g,ov.N_INIT,b)[0]
        p=lambda s:float((int((s>=o).sum())+1)/(ns+1));return dict(run=rn,i=i,seed=str(s),draw_seed=str(d),stat=st,pM=p(sM),pC=p(sC),obs=float(o),k=int(k))
    except Exception:return{"error":traceback.format_exc(),"run":rn,"i":i}
def _rel(p):
    try:return str(Path(p).relative_to(_R)).replace('\\',"/")
    except ValueError:return str(p)
def rate(c,n):return dict(events=int(c),n=int(n),rate=c/n,wilson95=list(nc._wilson(int(c),int(n))))
def main():
    wh=sys.argv[1] if len(sys.argv)>1 else "all"
    import scipy
    import sklearn
    ah=json.loads(ARCHIVE.read_text());cg=ah["config"];n,n1,ns,d=(cg["n"],cg["n_datasets"],cg["n_draws"],cg["deep_draws"]);t=ah["cells"]["nwb"]["tau"];l=ov.spec_exp(ov.D,t)
    s1=json.loads(OUT.read_text()) if OUT.exists() else{"cells":{}}
    s1["config"]=dict(experiment="within/between and Davies-Bouldin verification on disjoint seed blocks",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),n=n,d=ov.D,pr_target=cg["pr_target"],tau=t,n_datasets=n1,n_draws=ns,deep_draws=d,n_runs=N_RUNS,n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,statistics=cg["statistics"],archived=_rel(ARCHIVE),seed_rule="ov_common.hashed_seed(%d, %d, statistic index, run, dataset, purpose)"%(ROOT_SEED,EXPERIMENT_ID),versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS})
    ss=cg["statistics"] if wh=="all" else[wh];t0=time.time()
    for sd,st in enumerate(cg["statistics"]):
        if st not in ss or st in s1["cells"]:continue
        tc=time.time();a1=[]
        for rn in range(N_RUNS):
            for i in range(n1):a1.append((rn,i,ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,sd,rn,i,0),ov.hashed_seed(ROOT_SEED,EXPERIMENT_ID,sd,rn,i,1),ns,st,n))
        assert len({a[2] for a in a1}|{a[3] for a in a1})==2*N_RUNS*n1
        with Pool(N_WORKERS,initializer=_init,initargs=(l,))as pl:
            rs=list(pl.imap_unordered(one,a1,chunksize=1));e=[r for r in rs if "error" in r]
            if e:raise RuntimeError(e[0]["error"])
            r1=sorted((r for r in rs if "pM" in r),key=lambda r:(r["run"],r["i"]));dc=[r for r in r1 if(r["pM"]<ov.ALPHA)!=(r["pC"]<ov.ALPHA)]
            ds=list(pl.imap_unordered(one,[(r["run"],r["i"],int(r["seed"]),int(r["draw_seed"]),d,st,n) for r in dc],chunksize=1)) if dc else[]
        by={(r["run"],r["i"]):r for r in ds if "pM" in r};p=0
        for r in dc:
            g=by.get((r["run"],r["i"]));w=r["pC"]<ov.ALPHA
            if g is None:
                r["deep"]=None;continue
            nw=(g["pM"]<ov.ALPHA)!=(g["pC"]<ov.ALPHA);se=bool(nw and(g["pC"]<ov.ALPHA)==w);r["deep"]=dict(pM=g["pM"],pC=g["pC"],persists=se);p+=int(se and w)
        pM=np.array([r["pM"] for r in r1]);pC=np.array([r["pC"] for r in r1]);c=int(((pC<ov.ALPHA)&(pM>=ov.ALPHA)).sum());m=int(((pM<ov.ALPHA)&(pC>=ov.ALPHA)).sum());a=ah["cells"][st]
        s1["cells"][st]=dict(statistic=st,n_datasets=len(r1),M=rate(int((pM<ov.ALPHA).sum()),len(r1)),C=rate(int((pC<ov.ALPHA).sum()),len(r1)),C_only=c,M_only=m,runs=[dict(run=r,M_events=int(sum((1 for x in r1 if x["run"]==r and x["pM"]<ov.ALPHA))),C_events=int(sum((1 for x in r1 if x["run"]==r and x["pC"]<ov.ALPHA)))) for r in range(N_RUNS)],deep=dict(draws=d,tested=len(dc),C_only_persisting=p,persisting_fraction=p/c if c else None),archived=dict(pooled_M=a["pooled_M"],pooled_C=a["pooled_C"],total_C_only=a["total_C_only"],total_M_only=a["total_M_only"],deep=a["deep"]),discordant=dc,wall_s=round(time.time()-tc,1))
        OUT.write_text(json.dumps(s1,indent=1));s=s1["cells"][st]
        print("[%s] M %.4f %s  C %.4f %s  C-only %d  M-only %d  persist %d/%d  [%.0fs]"%(st,s["M"]["rate"],s["M"]["wilson95"],s["C"]["rate"],s["C"]["wilson95"],c,m,p,c,s["wall_s"]),flush=True)
    s1["config"]["total_wall_s"]=round(time.time()-t0,1);OUT.write_text(json.dumps(s1,indent=1));print("wrote %s"%OUT)
if __name__=="__main__":main()
