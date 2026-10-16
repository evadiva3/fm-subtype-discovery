#Phase 1D: regenerate the two one-dimensional sweeps of Section 3.2 and the
import json
import os
import platform
import sys
import time
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nullcal as nc
import ov_common as ov
D=119;N_DRAWS=200;N_INIT=5;K_GRID=(2,3,4,5,6);ALPHA=0.05;NDS_SWEEP=120;NDS_COMPANION=200;SEED_A=3100000;SEED_B=3200000;SEED_COMP=3300000;SWEEP_A_TAUS=[0.5,1.0,2.0,3.0,5.0,8.0,15.0,40.0,float("inf")]
SWEEP_B_NS=[28,50,100,200];SWEEP_B_TAU=3.0
def _mc(args):
    n,l,i,sd=args
    try:
        X=ov.gen(n,D,l,sd+1000*i);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(X,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
        if k is None:return{"skipped":True}
        kw=dict(k_grid=K_GRID,n_draws=N_DRAWS,min_size=g,n_init=N_INIT,seed=sd+1000*i+7,observed=(s,k));rM=nc.null_test(X,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(X,reproject=True,match_selection=True,**kw)
        return{"pr":float(nc.participation_ratio(X)),"obs":float(s),"k":int(k),"pM":float(rM.p),"pC":float(rC.p),"nullM":float(rM.null_mean),"nullC":float(rC.null_mean)}
    except Exception as e:return{"error":f"{type(e).__name__}: {e}"}
def _four(args):
    n,l,i,sd=args
    try:
        X=ov.gen(n,D,l,sd+1000*i);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(X,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
        if k is None:return{"skipped":True}
        kw=dict(k_grid=K_GRID,n_draws=N_DRAWS,min_size=g,n_init=N_INIT,seed=sd+1000*i+7,observed=(s,k));o={"pr":float(nc.participation_ratio(X)),"obs":float(s),"k":int(k)}
        for ne,rp,ms in(("M",False,False),("G",True,False),("S",False,True),("C",True,True)):
            r=nc.null_test(X,reproject=rp,match_selection=ms,**kw);o["p"+ne]=float(r.p);o["null"+ne]=float(r.null_mean)
        return o
    except Exception as e:return{"error":f"{type(e).__name__}: {e}"}
def summarize(res,arms):
    g=[r for r in res if "p"+arms[0] in r]
    cl={"n_datasets":len(g),"n_skipped":sum((1 for r in res if r.get("skipped"))),"n_errors":sum((1 for r in res if "error" in r)),"pr_mean":float(np.mean([r["pr"] for r in g])) if g else None,"pr_sd":float(np.std([r["pr"] for r in g],ddof=1)) if len(g)>1 else None,"pr_min":float(min((r["pr"] for r in g))) if g else None,"pr_max":float(max((r["pr"] for r in g))) if g else None,"obs_mean":float(np.mean([r["obs"] for r in g])) if g else None}
    for a in arms:
        p=np.array([r["p"+a] for r in g]);c=int(np.sum(p<ALPHA))
        cl[a]={"rej":float(c/p.size) if p.size else None,"events":c,"ci_wilson":list(nc._wilson(c,p.size)) if p.size else None,"mean_p":float(p.mean()) if p.size else None,"ks_D":float(nc._ks_uniform(p)) if p.size else None,"null_mean":float(np.mean([r["null"+a] for r in g])) if g else None}
    if "M" in arms and "C" in arms:
        pM=np.array([r["pM"] for r in g]);pC=np.array([r["pC"] for r in g]);cl["M_only"]=int(np.sum((pM<ALPHA)&(pC>=ALPHA)));cl["C_only"]=int(np.sum((pC<ALPHA)&(pM>=ALPHA)))
    return(cl,g)
def main():
    s=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");o1=ov.ROOT/"results"/"phase1"/s;o1.mkdir(parents=True,exist_ok=True);ot=o1/"r1d_geometry_sweeps.json"
    import sklearn
    import scipy
    h={"experiment":"R1D regenerated geometry sweeps","purpose":"replace the unarchived Sections 3.1 and 3.2 sweep artifacts","written_utc":s,"d":D,"n_draws":N_DRAWS,"n_init":N_INIT,"k_grid":list(K_GRID),"alpha":ALPHA,"n_datasets_sweep":NDS_SWEEP,"n_datasets_companion":NDS_COMPANION,"seed_base_sweep_A":SEED_A,"seed_base_sweep_B":SEED_B,"seed_base_companion":SEED_COMP,"seed_rule":"dataset i uses seed_base + 1000*i; its draws use seed_base + 1000*i + 7","sweep_A_taus":["inf" if np.isinf(t) else t for t in SWEEP_A_TAUS],"sweep_B_ns":SWEEP_B_NS,"sweep_B_tau":SWEEP_B_TAU,"generative_model":"single unimodal Gaussian, eigenvalues exp(-i/tau) in a random orthonormal basis, rows L2-normalized","versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform()}}
    se={"header":h,"sweep_A":{},"sweep_B":{},"companion":{}};tt=time.time()
    with Pool(47)as po:
        for tu in SWEEP_A_TAUS:
            k="tau_inf" if np.isinf(tu) else f"tau{tu:g}";l=ov.spec_exp(D,tu);t0=time.time();a=[(28,l,i,SEED_A+int(0 if np.isinf(tu) else tu*977)) for i in range(NDS_SWEEP)]
            r=list(po.imap_unordered(_mc,a,chunksize=1));c,g=summarize(r,("M","C"));c.update(tau=None if np.isinf(tu) else tu,n=28,wall_s=round(time.time()-t0,1));se["sweep_A"][k]=c
            print(f"[A] {k:9s} PR {c["pr_mean"]:.2f}  M {c["M"]["rej"]:.3f}  C {c["C"]["rej"]:.3f}  {c["wall_s"]:.0f}s",flush=True);json.dump(se,open(ot,"w"),indent=1)
        l=ov.spec_exp(D,SWEEP_B_TAU)
        for n in SWEEP_B_NS:
            t0=time.time();a=[(n,l,i,SEED_B+n*977) for i in range(NDS_SWEEP)];r=list(po.imap_unordered(_mc,a,chunksize=1));c,g=summarize(r,("M","C"))
            c.update(tau=SWEEP_B_TAU,n=n,n_over_d=round(n/D,4),wall_s=round(time.time()-t0,1));se["sweep_B"][f"n{n}"]=c
            print(f"[B] n={n:<4d}    PR {c["pr_mean"]:.2f}  M {c["M"]["rej"]:.3f}  C {c["C"]["rej"]:.3f}  {c["wall_s"]:.0f}s",flush=True);json.dump(se,open(ot,"w"),indent=1)
        tc,p,o,nc=ov.bisect_param(28,D,lambda t:ov.spec_exp(D,t),0.2,60.0,4.36,tol=0.02,m=200);l=ov.spec_exp(D,tc);t0=time.time();a=[(28,l,i,SEED_COMP) for i in range(NDS_COMPANION)]
        r=list(po.imap_unordered(_four,a,chunksize=1));c,g=summarize(r,("M","G","S","C"));c.update(tau=float(tc),tuned_pr=float(p),tune_ok=bool(o),tune_note=nc,n=28,wall_s=round(time.time()-t0,1))
        se["companion"]=c;print(f"[companion] tau {tc:.4f} tuned PR {p:.3f}  "+"  ".join((f"{a} {c[a]["rej"]:.3f}"for a in("M","G","S","C")))+f"  {c["wall_s"]:.0f}s",flush=True)
    se["header"]["total_wall_s"]=round(time.time()-tt,1);json.dump(se,open(ot,"w"),indent=1);print(f'\nwrote {ot}');print(f"total wall {se["header"]["total_wall_s"] / 60:.1f} min")
if __name__=="__main__":main()
