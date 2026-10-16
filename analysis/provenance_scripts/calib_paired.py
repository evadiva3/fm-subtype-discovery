from __future__ import annotations
import os
from _paths import RP,out
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import json,time,warnings
import multiprocessing as mp
import numpy as np

OUT=out("calibration_paired")

ND=int(os.environ.get("NDATASETS",1000))
NDR=int(os.environ.get("NDRAWS",1000))

def amb_null(x,rng):
    mu=x.mean(axis=0)
    cov=np.cov(x,rowvar=False)
    return rng.multivariate_normal(mu,cov,size=x.shape[0],method="svd")

def one_ds(arg):
    i,xn=arg
    warnings.simplefilter("ignore")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from analysis.evaluate import cluster_evaluate
    from config import config

    ev=cluster_evaluate()
    rng=np.random.default_rng(10_000+i)
    sy=ev._null_mvn(xn,rng)

    n=sy.shape[0]
    ms=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*n))
    sils,mins,labs=[],[],[]
    for k in config.kmeansKRange:
        lb=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=i).fit_predict(sy)
        sils.append(silhouette_score(sy,lb));mins.append(int(np.bincount(lb).min()));labs.append(lb)
    ok=[j for j in range(len(config.kmeansKRange)) if mins[j]>=ms]
    b=max(ok,key=lambda j:sils[j]) if ok else int(np.argmax(sils))
    lab,kSel,obs=labs[b],int(config.kmeansKRange[b]),float(sils[b])

    p_corr=ev.perm(sy,lab,n_permutations=NDR,random_state=i,match_selection=True)

    rng2=np.random.default_rng(50_000+i)
    c=0
    for j in range(NDR):
        nl=amb_null(sy,rng2)
        lb=KMeans(n_clusters=kSel,n_init=config.kmeansNInit,random_state=j).fit_predict(nl)
        if silhouette_score(nl,lb)>=obs:
            c+=1
    p_mis=(c+1)/(NDR+1)

    return float(p_corr),float(p_mis),kSel

def wilson(k,n,z=1.96):
    if n==0:return(0.0,0.0)
    ph=k/n;d=1+z*z/n
    c=(ph+z*z/(2*n))/d
    h=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return(max(0.0,c-h),min(1.0,c+h))

def main():
    from scipy import stats
    x=np.load(RP/"data/outputs/trained_fm_embeddings.npy").astype(float)
    xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    nw=min(ND,os.cpu_count() or 4)
    print(f"paired calibration: {ND} datasets x {NDR} draws x 2 tests, embeddings {x.shape}, {nw} workers",flush=True)

    t0=time.perf_counter()
    with mp.Pool(nw) as pool:
        res=pool.map(one_ds,[(i,xn) for i in range(ND)],chunksize=4)
    el=(time.perf_counter()-t0)/60

    pc=np.array([r[0] for r in res]);pm=np.array([r[1] for r in res])
    ks=np.array([r[2] for r in res])
    OUT.mkdir(parents=True,exist_ok=True)
    np.save(OUT/"p_corrected.npy",pc);np.save(OUT/"p_misspecified.npy",pm)
    np.save(OUT/"selected_k.npy",ks)

    sm={"n_datasets":ND,"n_draws":NDR,"elapsed_min":round(el,1),"embedding_shape":list(x.shape)}
    for nm,p in(("corrected",pc),("misspecified",pm)):
        r=int((p<0.05).sum());lo,hi=wilson(r,len(p))
        kst=stats.kstest(p,"uniform")
        sm[nm]={"mean_p":float(p.mean()),"median_p":float(np.median(p)),
                "rejections_at_05":r,"rejection_rate_05":r/len(p),
                "wilson95_lo":lo,"wilson95_hi":hi,
                "ks_D":float(kst.statistic),"ks_p":float(kst.pvalue)}
    nb=int(((pm<0.05)&~(pc<0.05)).sum());nc=int((~(pm<0.05)&(pc<0.05)).sum())
    sm["mcnemar_discordant"]={"mis_only":nb,"corr_only":nc,
                              "p":float(stats.binomtest(nb,nb+nc,0.5).pvalue) if nb+nc else None}
    sm["selected_k_counts"]={int(k):int((ks==k).sum()) for k in np.unique(ks)}
    (OUT/"calibration_paired_summary.json").write_text(json.dumps(sm,indent=2))

    print(f"\n  elapsed {el:.1f} min\n")
    print(f"  {'test':<14}{'mean p':>9}{'reject@.05':>12}{'95% CI':>20}{'KS D':>9}{'KS p':>11}")
    for nm in("corrected","misspecified"):
        s=sm[nm]
        ci="[{:.4f},{:.4f}]".format(s["wilson95_lo"],s["wilson95_hi"])
        print(f"  {nm:<14}{s['mean_p']:>9.4f}{s['rejection_rate_05']:>12.4f}{ci:>20}"
              f"{s['ks_D']:>9.4f}{s['ks_p']:>11.3e}")
    print("\n  nominal alpha=0.05")
    print(f"  discordant pairs: misspecified-only rejections {nb}, corrected-only {nc}")
    print(f"  null's selected-k distribution: {sm['selected_k_counts']}")
    print(f"  wrote {OUT}")

if __name__=="__main__":
    main()
