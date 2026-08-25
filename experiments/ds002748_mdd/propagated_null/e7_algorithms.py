import json,time,traceback,warnings
from multiprocessing import Pool
import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import ov_common as ov
import nullcal as nc
warnings.filterwarnings("ignore")
OUT=ov.OUTDIR/"e7_algorithms.json"
N=28
D=ov.D
PR_TARGET=4.4
NDS=1200
N_DRAWS=200
ALGS=["kmeans","ward","gmm","spectral"]
DEGENERATE=nc._DEGENERATE


def fit_labels(alg,X,k,rs):
    try:
        if alg=="kmeans":
            return KMeans(n_clusters=k,n_init=ov.N_INIT,
                random_state=rs).fit_predict(X)
        if alg=="ward":
            return AgglomerativeClustering(n_clusters=k,
                linkage="ward").fit_predict(X)
        if alg=="gmm":
            return GaussianMixture(n_components=k,covariance_type="full",
                reg_covar=1e-4,n_init=1,
                random_state=rs).fit_predict(X)
        if alg=="spectral":
            return SpectralClustering(n_clusters=k,affinity="rbf",
                assign_labels="kmeans",
                random_state=rs).fit_predict(X)
    except Exception:
        return None
    raise ValueError(alg)


def best_sil(alg,X,k_grid,min_size,rs):
    n=X.shape[0]
    b=(DEGENERATE,None)
    f=0
    for k in k_grid:
        if k<2 or k>=n:
            continue
        lb=fit_labels(alg,X,k,rs)
        if lb is None:
            f+=1
            continue
        if len(np.unique(lb))<2:
            continue
        if np.bincount(lb,minlength=k).min()<min_size:
            continue
        s=float(silhouette_score(X,lb))
        if s>b[0]:
            b=(s,int(k))
    return b[0],b[1],f


def sil_at_k(alg,X,k,rs):
    lb=fit_labels(alg,X,k,rs)
    if lb is None:
        return DEGENERATE,1
    if len(np.unique(lb))<2:
        return DEGENERATE,0
    return float(silhouette_score(X,lb)),0


def worker(args):
    n,d,lam,idx,s,nd=args
    try:
        Xn=ov.gen(n,d,lam,s)
        pr=nc.participation_ratio(Xn)
        ms=nc.min_cluster_guard(n)
        m,A=nc._gaussian_factor(Xn)
        out={"pr":float(pr)}
        for alg in ALGS:
            os,ok,f0=best_sil(alg,Xn,ov.K_GRID,ms,idx)
            if ok is None:
                out[alg]=None
                continue
            rn=np.random.default_rng(s+7919)
            r=np.random.default_rng(s+7919)
            sM=np.empty(nd)
            sC=np.empty(nd)
            f=f0
            for b in range(nd):
                Dm=nc._draw(m,A,n,rn)
                v,fa=sil_at_k(alg,Dm,ok,b)
                sM[b]=v
                f+=fa
                Dc=nc.l2_normalize(nc._draw(m,A,n,r))
                v2,_,fb=best_sil(alg,Dc,ov.K_GRID,ms,b)
                sC[b]=v2
                f+=fb
            cM=int((sM>=os).sum())
            cC=int((sC>=os).sum())
            out[alg]=dict(pM=float((cM+1)/(nd+1)),
                pC=float((cC+1)/(nd+1)),
                obs=float(os),k=int(ok),fails=int(f),
                nmM=float(sM.mean()),nmC=float(sC.mean()),
                nsM=float(sM.std(ddof=1)),nsC=float(sC.std(ddof=1)))
        return out
    except Exception:
        return{"error":traceback.format_exc()}


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        algorithms=ALGS,gmm_reg_covar=1e-4)
    s=ov.load_state(OUT,"E7",cfg)
    if s["cells"].get("done"):
        print("E7 already complete")
        return
    t0=time.time()
    tau,pt,ok,no=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E7] tau=%.4f tuned PR=%.3f ok=%s"%(tau,pt,ok),flush=True)
    ar=[(N,D,lam,i,200_000+1000*i,N_DRAWS)for i in range(NDS)]
    with Pool(47)as po:
        res=list(po.imap_unordered(worker,ar,chunksize=1))
    g=[r for r in res if "pr" in r]
    e=[r["error"]for r in res if "error" in r]
    a=ov.ALPHA
    ce={}
    for alg in ALGS:
        ro=[r[alg]for r in g if r.get(alg)]
        if not ro:
            ce[alg]=dict(algorithm=alg,failed=True,
                n_no_valid_k=sum(1 for r in g if r.get(alg)is None))
            continue
        pM=np.array([x["pM"]for x in ro])
        pC=np.array([x["pC"]for x in ro])
        cM,cC,Nn=int((pM<a).sum()),int((pC<a).sum()),len(ro)
        ce[alg]=dict(
            algorithm=alg,n_datasets=Nn,n_draws=N_DRAWS,
            n_no_valid_k=sum(1 for r in g if r.get(alg)is None),
            fit_failures=int(sum(x["fails"]for x in ro)),
            rej_M=cM/Nn,count_M=cM,ci_M=list(nc._wilson(cM,Nn)),
            rej_C=cC/Nn,count_C=cC,ci_C=list(nc._wilson(cC,Nn)),
            mean_pM=float(pM.mean()),mean_pC=float(pC.mean()),
            ks_M=float(nc._ks_uniform(pM)),ks_C=float(nc._ks_uniform(pC)),
            null_mean_M=float(np.mean([x["nmM"]for x in ro])),
            null_mean_C=float(np.mean([x["nmC"]for x in ro])),
            obs_mean=float(np.mean([x["obs"]for x in ro])),
            M_only=int(((pM<a)&(pC>=a)).sum()),
            C_only=int(((pC<a)&(pM>=a)).sum()),
            gap=float(cM/Nn-cC/Nn),
            k_counts={int(k):int(sum(1 for x in ro if x["k"]==k))
            for k in sorted({x["k"]for x in ro})},
            failed=False)
    s["cells"]=ce
    s["meta"]=dict(tau=float(tau),tuned_pr=pt,tuned_ok=ok,
        pr_mean=float(np.mean([r["pr"]for r in g])),
        n_errors=len(e),wall_s=round(time.time()-t0,1),
        error_sample=e[0]if e else None)
    ov.save_state(OUT,s)
    for alg,c in ce.items():
        if c.get("failed"):
            print("[E7] %-9s FAILED"%alg,flush=True)
            continue
        f="  <<< C_only>0" if c["C_only"]>0 else ""
        print("[E7] %-9s M=%.4f(%d) C=%.4f(%d) gap=%.4f Monly=%d Conly=%d "
            "fitfail=%d%s"%(alg,c["rej_M"],c["count_M"],c["rej_C"],
            c["count_C"],c["gap"],c["M_only"],c["C_only"],
            c["fit_failures"],f),flush=True)
    print("E7 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
