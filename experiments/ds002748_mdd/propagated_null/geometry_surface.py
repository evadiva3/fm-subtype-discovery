import json
import os
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
from multiprocessing import Pool
import nullcal as nc
N_DRAWS=200
N_DATASETS=120
N_INIT=5
K_GRID=(2,3,4,5,6)
ALPHA=0.05
OUT="geometry_surface.json"
def make_cov_basis(d:int,tau:float,rng:np.random.Generator):
    if np.isinf(tau):
        lam=np.ones(d)
    else:
        lam=np.exp(-np.arange(d)/tau)
    Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    return np.sqrt(lam)[:,None]*Q.T


def one_dataset(args):
    n,d,tau,idx,s=args
    rng=np.random.default_rng(s)
    A=make_cov_basis(d,tau,rng)
    X=rng.standard_normal((n,d))@A
    Xn=nc.l2_normalize(X)
    ps=nc.participation_ratio(Xn)
    obs=nc.best_silhouette(Xn,k_grid=K_GRID,n_init=N_INIT,random_state=idx)
    if obs[1]is None:
        return None
    kw=dict(
        k_grid=K_GRID,
        n_draws=N_DRAWS,
        n_init=N_INIT,
        seed=s+7919,
        observed=(obs[0],obs[1]),
        )
    pM=nc.null_test(Xn,reproject=False,match_selection=False,**kw).p
    pC=nc.null_test(Xn,reproject=True,match_selection=True,**kw).p
    return dict(pr=float(ps),pM=float(pM),pC=float(pC),k=int(obs[1]))


def run_cell(n,d,tau,base_seed,pool):
    a=[(n,d,tau,i,base_seed+1000*i)for i in range(N_DATASETS)]
    res=[r for r in pool.imap_unordered(one_dataset,a,chunksize=2)if r]
    pM=np.array([r["pM"]for r in res])
    pC=np.array([r["pC"]for r in res])
    pr=np.array([r["pr"]for r in res])
    rM=float(np.mean(pM<ALPHA))
    rC=float(np.mean(pC<ALPHA))
    return dict(
        n=n,
        d=d,
        tau=None if np.isinf(tau)else float(tau),
        n_datasets=len(res),
        pr_mean=float(pr.mean()),
        pr_sd=float(pr.std(ddof=1)),
        rej_M=rM,
        ci_M=list(nc._wilson(int(np.sum(pM<ALPHA)),len(res))),
        rej_C=rC,
        ci_C=list(nc._wilson(int(np.sum(pC<ALPHA)),len(res))),
        M_only=int(np.sum((pM<ALPHA)&(pC>=ALPHA))),
        C_only=int(np.sum((pC<ALPHA)&(pM>=ALPHA))),
        mean_pM=float(pM.mean()),
        mean_pC=float(pC.mean()),
        ks_M=float(nc._ks_uniform(pM)),
        ks_C=float(nc._ks_uniform(pC)),
        )


def main():
    t0=time.time()
    r={"sweep_A_anisotropy":[],"sweep_B_sample_size":[],"config":{
        "n_draws":N_DRAWS,"n_datasets":N_DATASETS,"n_init":N_INIT,
        "k_grid":list(K_GRID),"alpha":ALPHA}}
    with Pool(2)as p:
        for tau in[0.5,1.0,2.0,3.0,5.0,8.0,15.0,40.0,np.inf]:
            c=run_cell(28,119,tau,base_seed=101,pool=p)
            r["sweep_A_anisotropy"].append(c)
            print(
                f"[A] tau={c['tau']}  PR={c['pr_mean']:.2f}  "
                f"M={c['rej_M']:.3f}  C={c['rej_C']:.3f}  "
                f"M-only={c['M_only']} C-only={c['C_only']}  "
                f"({time.time()-t0:.0f}s)",flush=True)
            json.dump(r,open(OUT,"w"),indent=1)
        for n in[28,50,100,200]:
            c=run_cell(n,119,3.0,base_seed=202,pool=p)
            r["sweep_B_sample_size"].append(c)
            print(
                f"[B] n={n}  PR={c['pr_mean']:.2f}  "
                f"M={c['rej_M']:.3f}  C={c['rej_C']:.3f}  "
                f"M-only={c['M_only']} C-only={c['C_only']}  "
                f"({time.time()-t0:.0f}s)",flush=True)
            json.dump(r,open(OUT,"w"),indent=1)
    print(f"done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
