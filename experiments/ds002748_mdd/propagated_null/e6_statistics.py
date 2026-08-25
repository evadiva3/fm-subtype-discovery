import json,time,traceback
from multiprocessing import Pool
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import(silhouette_score,calinski_harabasz_score,
    davies_bouldin_score)
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e6_statistics.json"
N=28
D=ov.D
PR_TARGET=4.4
NDS=1200
N_DRAWS=200
STATS=["silhouette","ch","ndb","nwb"]
DEGENERATE=-1e18


def stat_value(name,X,labels):
    if name=="silhouette":
        return float(silhouette_score(X,labels))
    if name=="ch":
        return float(calinski_harabasz_score(X,labels))
    if name=="ndb":
        return float(-davies_bouldin_score(X,labels))
    if name=="nwb":
        g=X.mean(axis=0)
        W=0.0
        B=0.0
        for u in np.unique(labels):
            P=X[labels==u]
            c=P.mean(axis=0)
            W+=float(((P-c)**2).sum())
            B+=float(P.shape[0]*((c-g)**2).sum())
        if B<=0:
            return DEGENERATE
        return float(-(W/B))
    raise ValueError(name)


def best_stat(name,X,k_grid,min_size,n_init,random_state):
    n=X.shape[0]
    b=(DEGENERATE,None)
    for k in k_grid:
        if k<2 or k>=n:
            continue
        lb=KMeans(n_clusters=k,n_init=n_init,
            random_state=random_state).fit_predict(X)
        if len(np.unique(lb))<2:
            continue
        if np.bincount(lb,minlength=k).min()<min_size:
            continue
        s=stat_value(name,X,lb)
        if s>b[0]:
            b=(s,int(k))
    return b


def stat_at_k(name,X,k,n_init,random_state):
    lb=KMeans(n_clusters=k,n_init=n_init,
        random_state=random_state).fit_predict(X)
    if len(np.unique(lb))<2:
        return DEGENERATE
    return stat_value(name,X,lb)


def worker(args):
    n,d,lam,idx,s,nd=args
    try:
        Xn=ov.gen(n,d,lam,s)
        pr=nc.participation_ratio(Xn)
        ms=nc.min_cluster_guard(n)
        m,A=nc._gaussian_factor(Xn)
        out={"pr":float(pr)}
        for na in STATS:
            os,ok=best_stat(na,Xn,ov.K_GRID,ms,ov.N_INIT,idx)
            if ok is None:
                out[na]=None
                continue
            rn=np.random.default_rng(s+7919)
            r=np.random.default_rng(s+7919)
            sM=np.empty(nd)
            sC=np.empty(nd)
            for b in range(nd):
                Dm=nc._draw(m,A,n,rn)
                sM[b]=stat_at_k(na,Dm,ok,ov.N_INIT,b)
                Dc=nc.l2_normalize(nc._draw(m,A,n,r))
                sC[b]=best_stat(na,Dc,ov.K_GRID,ms,ov.N_INIT,b)[0]
            cM=int((sM>=os).sum())
            cC=int((sC>=os).sum())
            out[na]=dict(pM=float((cM+1)/(nd+1)),
                pC=float((cC+1)/(nd+1)),
                obs=float(os),k=int(ok),
                nmM=float(sM.mean()),nsM=float(sM.std(ddof=1)),
                nmC=float(sC.mean()),nsC=float(sC.std(ddof=1)))
        return out
    except Exception:
        return{"error":traceback.format_exc()}


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        statistics=STATS)
    s=ov.load_state(OUT,"E6",cfg)
    if s["cells"].get("done"):
        print("E6 already complete")
        return
    t0=time.time()
    tau,pt,ok,no=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E6] tau=%.4f tuned PR=%.3f ok=%s"%(tau,pt,ok),flush=True)
    ar=[(N,D,lam,i,100_000+1000*i,N_DRAWS)for i in range(NDS)]
    with Pool(47)as po:
        res=list(po.imap_unordered(worker,ar,chunksize=1))
    g=[r for r in res if "pr" in r]
    e=[r["error"]for r in res if "error" in r]
    ce={}
    a=ov.ALPHA
    for na in STATS:
        ro=[r[na]for r in g if r.get(na)]
        if not ro:
            ce[na]=dict(failed=True)
            continue
        pM=np.array([x["pM"]for x in ro])
        pC=np.array([x["pC"]for x in ro])
        cM,cC,Nn=int((pM<a).sum()),int((pC<a).sum()),len(ro)
        ce[na]=dict(
            statistic=na,n_datasets=Nn,n_draws=N_DRAWS,
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
    for na,c in ce.items():
        if c.get("failed"):
            print("[E6] %-11s FAILED"%na,flush=True)
            continue
        f="  <<< C_only>0" if c["C_only"]>0 else ""
        print("[E6] %-11s M=%.4f(%d) [%.3f,%.3f]  C=%.4f(%d) [%.3f,%.3f]  "
            "gap=%.4f Monly=%d Conly=%d%s"
            %(na,c["rej_M"],c["count_M"],c["ci_M"][0],c["ci_M"][1],
            c["rej_C"],c["count_C"],c["ci_C"][0],c["ci_C"][1],
            c["gap"],c["M_only"],c["C_only"],f),flush=True)
    print("E6 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
