import json,glob,csv,time
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score,silhouette_score
import ov_common as ov
import nullcal as nc
from e11_consensus import load_fixed_arch,load_search_vary,cooccurrence,consensus_partition
OUT=ov.OUTDIR/"e17_consensus_null.json"
N_SURROGATE=300
N_BOOT_OBS=1000
N_BOOT_SUR=100
SEED=20260822

def stability(L,guard,n_boot,rng):
    nrun=L.shape[0]
    co=cooccurrence(L)
    c,k,sil,_=consensus_partition(co,guard)
    if c is None:
        return None
    v=[]
    for _ in range(n_boot):
        idx=rng.integers(0,nrun,size=nrun)
        cb,_,_,_=consensus_partition(cooccurrence(L[idx]),guard)
        if cb is None:
            continue
        v.append(adjusted_rand_score(cb,c))
    p=[adjusted_rand_score(L[i],L[j]) for i in range(nrun) for j in range(i+1,nrun)]
    return dict(consensus_k=int(k),consensus_silhouette=float(sil),boot_mean=float(np.mean(v)),boot_sd=float(np.std(v,ddof=1)),pairwise_mean=float(np.mean(p)),run_vs_consensus_mean=float(np.mean([adjusted_rand_score(L[r],c) for r in range(nrun)])))

def main():
    t0=time.time()
    st=ov.load_state(OUT,"E17",dict(n_surrogate=N_SURROGATE,n_boot_observed=N_BOOT_OBS,n_boot_surrogate=N_BOOT_SUR,seed=SEED))
    for na,fn in (("fixed_arch",load_fixed_arch), ("search_vary",load_search_vary)):
        if na in st["cells"]:
            print("  skip %s"%na,flush=True)
            continue
        labs,ids,nf=fn()
        if labs is None:
            st["cells"][na]=dict(protocol=na,skipped=True)
            continue
        nrun,n=labs.shape
        g=nc.min_cluster_guard(n)
        rng=np.random.default_rng(SEED)
        obs=stability(labs,g,N_BOOT_OBS,rng)
        print("[E17] %-12s observed: consensus k=%d  bootstrap ARI %.4f +- %.4f  pairwise %.4f"% (na,obs["consensus_k"],obs["boot_mean"],obs["boot_sd"],obs["pairwise_mean"]),flush=True)
        sur=[]
        for s in range(N_SURROGATE):
            r2=np.random.default_rng(SEED+1000+s)
            Ls=np.stack([r2.permutation(labs[r]) for r in range(nrun)])
            v=stability(Ls,g,N_BOOT_SUR,r2)
            if v:
                sur.append(v)
            if (s+1)%100==0:
                print("        surrogate %d/%d"% (s+1,N_SURROGATE),flush=True)
        sb=np.array([v["boot_mean"] for v in sur])
        sp=np.array([v["pairwise_mean"] for v in sur])
        sk=[v["consensus_k"] for v in sur]
        p=float((int((sb>=obs["boot_mean"]).sum())+1)/(len(sb)+1))
        c=dict(protocol=na,n_runs=nrun,n_subjects=n,guard=g,observed=obs,n_surrogate=len(sur),surrogate_boot_mean=float(sb.mean()),surrogate_boot_sd=float(sb.std(ddof=1)),surrogate_boot_q05=float(np.quantile(sb,0.05)),surrogate_boot_q95=float(np.quantile(sb,0.95)),surrogate_boot_max=float(sb.max()),surrogate_pairwise_mean=float(sp.mean()),surrogate_consensus_k_counts={int(k):int(sk.count(k)) for k in sorted(set(sk))},p_value=p,excess=float(obs["boot_mean"]-sb.mean()),wall_s=round(time.time()-t0,1))
        st["cells"][na]=c
        ov.save_state(OUT,st)
        print("[E17] %-12s surrogate bootstrap ARI %.4f +- %.4f (95th %.4f, max %.4f)"% (na,sb.mean(),sb.std(ddof=1),np.quantile(sb,0.95),sb.max()),flush=True)
        print("[E17] %-12s surrogate pairwise ARI %.4f   ->  observed exceeds surrogate by %+.4f, permutation p = %.4f"% (na,sp.mean(),c["excess"],p),flush=True)
    print("E17 done in %.0fs"% (time.time()-t0))
if __name__=="__main__":
    main()
