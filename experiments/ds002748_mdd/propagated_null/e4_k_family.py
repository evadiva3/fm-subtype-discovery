import json,time,traceback
from multiprocessing import Pool
import numpy as np
from sklearn.cluster import KMeans
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e4_k_family.json"
CSV=ov.OUTDIR/"e4_k_family.csv"
N=28
D=ov.D
PR_TARGET=4.4
NDS=400
N_DRAWS=200
KMAX=11
N_GUARD_PROBE=20

CSV_COLS=["cell","k_max","nominal_K","eff_K_observed","eff_K_null",
    "pr_mean","n_datasets","n_draws",
    "rej_M","count_M","ci_M_lo","ci_M_hi",
    "rej_C","count_C","ci_C_lo","ci_C_hi",
    "mean_pM","mean_pC","ks_M","ks_C",
    "null_mean_M","null_mean_C","M_only","C_only","wall_s"]


def n_pass_guard(X,k_grid,min_size,n_init,rs):
    c=0
    for k in k_grid:
        if k<2 or k>=X.shape[0]:
            continue
        lb=KMeans(n_clusters=k,n_init=n_init,random_state=rs).fit_predict(X)
        if len(np.unique(lb))>=2 and np.bincount(lb,minlength=k).min()>=min_size:
            c+=1
    return c


def worker(args):
    n,d,lam,idx,s,kg,nd=args
    try:
        Xn=ov.gen(n,d,lam,s)
        pr=nc.participation_ratio(Xn)
        ms=nc.min_cluster_guard(n)
        eo=n_pass_guard(Xn,kg,ms,ov.N_INIT,idx)
        obs=nc.best_silhouette(Xn,k_grid=kg,min_size=ms,
            n_init=ov.N_INIT,random_state=idx)
        if obs[1]is None:
            return{"skip":"no k passes guard on observed"}
        m,A=nc._gaussian_factor(Xn)
        rng=np.random.default_rng(s+4242)
        en=float(np.mean([
            n_pass_guard(nc.l2_normalize(nc._draw(m,A,n,rng)),
            kg,ms,ov.N_INIT,b)
            for b in range(N_GUARD_PROBE)]))
        kw=dict(k_grid=kg,n_draws=nd,min_size=ms,n_init=ov.N_INIT,
            seed=s+7919,observed=(obs[0],obs[1]))
        rM=nc.null_test(Xn,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,**kw)
        return dict(pr=float(pr),pM=float(rM.p),pC=float(rC.p),k=int(obs[1]),
            obs=float(obs[0]),eff_obs=int(eo),eff_null=en,
            nmM=float(rM.null_mean),nsM=float(rM.null_sd),
            nmC=float(rC.null_mean),nsC=float(rC.null_sd),
            degC=int(rC.n_degenerate))
    except Exception:
        return{"error":traceback.format_exc()}


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,alpha=ov.ALPHA,k_max_range=list(range(2,KMAX+1)),
        guard=nc.min_cluster_guard(N),guard_probe_draws=N_GUARD_PROBE)
    s=ov.load_state(OUT,"E4",cfg)
    t0=time.time()
    tau,pt,ok,no=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E4] tau=%.4f tuned PR=%.3f ok=%s %s  guard=%d"
        %(tau,pt,ok,no,nc.min_cluster_guard(N)),flush=True)
    with Pool(47)as po:
        for k in range(2,KMAX+1):
            key="K2_%d"%k
            if key in s["cells"]:
                print("  skip %s"%key,flush=True)
                continue
            tc=time.time()
            kg=tuple(range(2,k+1))
            a=[(N,D,lam,i,600_000+1000*i+k,kg,N_DRAWS)
                for i in range(NDS)]
            res=list(po.imap_unordered(worker,a,chunksize=1))
            g=[r for r in res if "pM" in r]
            c=ov.summarize_mc(
                res,time.time()-tc,k_max=k,nominal_K=len(kg),
                k_grid=list(kg),tau=float(tau),n=N,d=D,n_draws=N_DRAWS,
                eff_K_observed=float(np.mean([r["eff_obs"]for r in g]))if g else None,
                eff_K_null=float(np.mean([r["eff_null"]for r in g]))if g else None)
            s["cells"][key]=c
            ov.save_state(OUT,s)
            ov.write_csv(CSV,s["cells"],CSV_COLS)
            if c.get("failed"):
                print("[E4] %-8s FAILED errors=%d"%(key,c["n_errors"]),flush=True)
            else:
                f="  <<< C_only>0" if c["C_only"]>0 else ""
                print("[E4] K={2..%2d} nom|K|=%2d effK_obs=%.2f effK_null=%.2f "
                    "M=%.3f(%d) C=%.3f(%d) Monly=%d Conly=%d [%.0fs tot %.0fs]%s"
                    %(k,len(kg),c["eff_K_observed"],c["eff_K_null"],
                    c["rej_M"],c["count_M"],c["rej_C"],c["count_C"],
                    c["M_only"],c["C_only"],c["wall_s"],
                    time.time()-t0,f),flush=True)
    print("E4 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
