import os,sys,json,time,warnings,traceback
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_v]="1"
import numpy as np
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nullcal as nc
ROOT=Path(__file__).resolve().parents[3]
OUTDIR=ROOT/"results"/"overnight"
OUT=OUTDIR/"e1_grid.json"
CSV=OUTDIR/"e1_grid.csv"
D=119
TARGET_PRS=[2.0,3.0,4.5,6.0,9.0,15.0]
NS=[28,60,119,180,240]
NDS={28:400,60:400,119:400,180:400,240:400}
N_DRAWS=200
N_INIT=5
K_GRID=(2,3,4,5,6)
ALPHA=0.05
TOL=0.05
TUNE_M=20


def cov_factor(d,tau,rng):
    lam=np.ones(d)if np.isinf(tau)else np.exp(-np.arange(d)/tau)
    Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    return np.sqrt(lam)[:,None]*Q.T


def mean_pr(n,d,tau,seed=987654,m=TUNE_M):
    prs=[]
    for i in range(m):
        rng=np.random.default_rng(seed+31*i)
        X=rng.standard_normal((n,d))@cov_factor(d,tau,rng)
        prs.append(nc.participation_ratio(nc.l2_normalize(X)))
    return float(np.mean(prs))


def tune_tau(n,d,target):
    lo,hi=0.02,1e5
    pl,ph=mean_pr(n,d,lo),mean_pr(n,d,hi)
    if target<=pl:
        return lo,pl,False,"target below floor at this n"
    if target>=ph:
        return hi,ph,False,"target above isotropic ceiling at this n"
    tau,pm=hi,ph
    for _ in range(60):
        tau=float(np.sqrt(lo*hi))
        pm=mean_pr(n,d,tau)
        if abs(pm-target)/target<=TOL:
            return tau,pm,True,""
        if pm<target:
            lo=tau
        else:
            hi=tau
        if hi/lo<1.0000001:
            break
    ok=abs(pm-target)/target<=TOL
    return tau,pm,ok,"" if ok else "bisection stalled"


def one(args):
    n,d,tau,idx,s=args
    try:
        rng=np.random.default_rng(s)
        X=rng.standard_normal((n,d))@cov_factor(d,tau,rng)
        Xn=nc.l2_normalize(X)
        pr=nc.participation_ratio(Xn)
        ms=nc.min_cluster_guard(n)
        obs=nc.best_silhouette(Xn,k_grid=K_GRID,min_size=ms,
            n_init=N_INIT,random_state=idx)
        if obs[1]is None:
            return{"skip":"no k passes guard on observed"}
        kw=dict(k_grid=K_GRID,n_draws=N_DRAWS,min_size=ms,n_init=N_INIT,
            seed=s+7919,observed=(obs[0],obs[1]))
        rM=nc.null_test(Xn,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,**kw)
        return dict(pr=float(pr),pM=float(rM.p),pC=float(rC.p),k=int(obs[1]),
            obs=float(obs[0]),
            nmM=float(rM.null_mean),nsM=float(rM.null_sd),
            nmC=float(rC.null_mean),nsC=float(rC.null_sd),
            degC=int(rC.n_degenerate))
    except Exception:
        return{"error":traceback.format_exc()}


def summarize(n,d,tau,pr_target,pr_tuned,tuned_ok,note,res,wall,nds_req):
    e=[r["error"]for r in res if "error" in r]
    s=[r for r in res if "skip" in r]
    g=[r for r in res if "pM" in r]
    if not g:
        return dict(n=n,d=d,n_over_d=n/d,pr_target=pr_target,tau=tau,
            failed=True,n_errors=len(e),n_skipped=len(s),
            error_sample=e[0]if e else None,wall_s=wall)
    pM=np.array([r["pM"]for r in g])
    pC=np.array([r["pC"]for r in g])
    pr=np.array([r["pr"]for r in g])
    cM=int(np.sum(pM<ALPHA))
    cC=int(np.sum(pC<ALPHA))
    N=len(g)
    return dict(
        n=n,d=d,n_over_d=round(n/d,4),
        pr_target=pr_target,tau=float(tau),
        pr_tuned_mean=pr_tuned,tau_tuned_ok=tuned_ok,tune_note=note,
        pr_mean=float(pr.mean()),pr_sd=float(pr.std(ddof=1)),
        n_datasets_requested=nds_req,n_datasets=N,n_draws=N_DRAWS,
        n_skipped=len(s),n_errors=len(e),
        rej_M=cM/N,count_M=cM,ci_M=list(nc._wilson(cM,N)),
        rej_C=cC/N,count_C=cC,ci_C=list(nc._wilson(cC,N)),
        mean_pM=float(pM.mean()),mean_pC=float(pC.mean()),
        ks_M=float(nc._ks_uniform(pM)),ks_C=float(nc._ks_uniform(pC)),
        null_mean_M=float(np.mean([r["nmM"]for r in g])),
        null_sd_M=float(np.mean([r["nsM"]for r in g])),
        null_mean_C=float(np.mean([r["nmC"]for r in g])),
        null_sd_C=float(np.mean([r["nsC"]for r in g])),
        obs_mean=float(np.mean([r["obs"]for r in g])),
        M_only=int(np.sum((pM<ALPHA)&(pC>=ALPHA))),
        C_only=int(np.sum((pC<ALPHA)&(pM>=ALPHA))),
        degenerate_C=int(np.sum([r["degC"]for r in g])),
        wall_s=round(wall,1),failed=False,
        )

CSV_COLS=["cell","n","d","n_over_d","pr_target","tau","pr_mean","pr_sd",
    "n_datasets","n_draws","rej_M","count_M","ci_M_lo","ci_M_hi",
    "rej_C","count_C","ci_C_lo","ci_C_hi","mean_pM","mean_pC",
    "ks_M","ks_C","null_mean_M","null_sd_M","null_mean_C","null_sd_C",
    "M_only","C_only","wall_s"]


def write_csv(cells):
    r=[",".join(CSV_COLS)]
    for key,c in cells.items():
        if c.get("failed"):
            continue
        r.append(",".join(str(x)for x in[
            key,c["n"],c["d"],c["n_over_d"],c["pr_target"],round(c["tau"],5),
            round(c["pr_mean"],4),round(c["pr_sd"],4),c["n_datasets"],c["n_draws"],
            c["rej_M"],c["count_M"],round(c["ci_M"][0],5),round(c["ci_M"][1],5),
            c["rej_C"],c["count_C"],round(c["ci_C"][0],5),round(c["ci_C"][1],5),
            round(c["mean_pM"],5),round(c["mean_pC"],5),
            round(c["ks_M"],5),round(c["ks_C"],5),
            round(c["null_mean_M"],5),round(c["null_sd_M"],5),
            round(c["null_mean_C"],5),round(c["null_sd_C"],5),
            c["M_only"],c["C_only"],c["wall_s"]]))
    CSV.write_text("\n".join(r)+"\n")


def main():
    OUTDIR.mkdir(parents=True,exist_ok=True)
    s=json.loads(OUT.read_text())if OUT.exists()else{
        "experiment":"E1","config":{
        "d":D,"target_prs":TARGET_PRS,"ns":NS,"n_datasets":NDS,
        "n_draws":N_DRAWS,"n_init":N_INIT,"k_grid":list(K_GRID),
        "alpha":ALPHA,"pr_tolerance":TOL,"tune_samples":TUNE_M},
        "cells":{}}
    t0=time.time()
    with Pool(47)as p:
        for pt in TARGET_PRS:
            for n in NS:
                key="pr%s_n%d"%(pt,n)
                if key in s["cells"]:
                    print("  skip %s (already present)"%key,flush=True)
                    continue
                tc=time.time()
                tau,prt,ok,no=tune_tau(n,D,pt)
                nds=NDS[n]
                off=int(round(pt*10))*1000+n
                a=[(n,D,tau,i,900_000+1000*i+off)
                    for i in range(nds)]
                res=list(p.imap_unordered(one,a,chunksize=1))
                c=summarize(n,D,tau,pt,prt,ok,no,
                    res,time.time()-tc,nds)
                s["cells"][key]=c
                OUT.write_text(json.dumps(s,indent=1))
                write_csv(s["cells"])
                if c.get("failed"):
                    print("[E1] %-14s FAILED errors=%d (%.0fs)"
                        %(key,c["n_errors"],time.time()-t0),flush=True)
                else:
                    f="  <<< C_only>0" if c["C_only"]>0 else ""
                    m="" if ok else "!"
                    print("[E1] %-14s tau=%9.4f PR=%6.2f(t%s%s) n/d=%.2f "
                        "M=%.3f(%d) C=%.3f(%d) Monly=%d Conly=%d [%.0fs tot %.0fs]%s"
                        %(key,tau,c["pr_mean"],pt,m,c["n_over_d"],
                        c["rej_M"],c["count_M"],c["rej_C"],
                        c["count_C"],c["M_only"],c["C_only"],
                        c["wall_s"],time.time()-t0,f),flush=True)
    print("E1 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
