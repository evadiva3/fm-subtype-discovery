import json,time,traceback,warnings
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
warnings.filterwarnings("ignore")
OUT=ov.OUTDIR/"e8_normalization.json"
N=28
D=ov.D
PR_TARGET=4.4
NDS=400
N_DRAWS=200
SCHEMES=["none","l2","zscore","whiten","pca_whiten"]
CONSTRUCTIONS={"M":(False,False),"G":(True,False),
    "S":(False,True),"C":(True,True)}


def t_none(X):
    return np.asarray(X,dtype=np.float64)


def t_l2(X):
    return nc.l2_normalize(X)


def t_zscore(X):
    X=np.asarray(X,dtype=np.float64)
    mu=X.mean(axis=0,keepdims=True)
    sd=X.std(axis=0,ddof=1,keepdims=True)
    return(X-mu)/np.maximum(sd,1e-12)


def _svd_parts(X):
    Xc=X-X.mean(axis=0,keepdims=True)
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    tol=max(Xc.shape)*np.finfo(float).eps*(S[0]if S.size else 0.0)
    r=int((S>tol).sum())
    return Xc,U,S,Vt,r


def t_whiten(X):
    X=np.asarray(X,dtype=np.float64)
    Xc,U,S,Vt,r=_svd_parts(X)
    if r==0:
        return Xc
    n=X.shape[0]
    W=(Vt[:r].T*(np.sqrt(n-1)/S[:r]))@Vt[:r]
    return Xc@W


def t_pca_whiten(X):
    X=np.asarray(X,dtype=np.float64)
    Xc,U,S,Vt,r=_svd_parts(X)
    if r==0:
        return Xc
    n=X.shape[0]
    return U[:,:r]*np.sqrt(n-1)

TRANSFORMS={"none":t_none,"l2":t_l2,"zscore":t_zscore,
    "whiten":t_whiten,"pca_whiten":t_pca_whiten}


def null_p(Xt,transform,reproject,match_selection,obs_stat,obs_k,
    k_grid,n_draws,min_size,n_init,seed):
    n=Xt.shape[0]
    m,A=nc._gaussian_factor(Xt)
    rng=np.random.default_rng(seed)
    st=np.empty(n_draws)
    for b in range(n_draws):
        Dd=nc._draw(m,A,n,rng)
        if reproject:
            Dd=transform(Dd)
        if match_selection:
            s,ks,_=nc.best_silhouette(Dd,k_grid=k_grid,min_size=min_size,
                n_init=n_init,random_state=b)
        else:
            s=nc.silhouette_at_k(Dd,obs_k,n_init=n_init,random_state=b)
        st[b]=s
    c=int((st>=obs_stat).sum())
    return(c+1)/(n_draws+1),float(st.mean()),float(st.std(ddof=1))


def worker(args):
    n,d,lam,idx,s,nd=args
    try:
        rng=np.random.default_rng(s)
        X=rng.standard_normal((n,d))@ov.factor(d,lam,rng)
        ms=nc.min_cluster_guard(n)
        out={}
        for sch in SCHEMES:
            tf=TRANSFORMS[sch]
            Xt=tf(X)
            pr=nc.participation_ratio(Xt)
            os,ok,_=nc.best_silhouette(Xt,k_grid=ov.K_GRID,
                min_size=ms,n_init=ov.N_INIT,
                random_state=idx)
            if ok is None:
                out[sch]=None
                continue
            rec={"pr":float(pr),"obs":float(os),"k":int(ok)}
            for c,(rep,m)in CONSTRUCTIONS.items():
                p,nm,nsd=null_p(Xt,tf,rep,m,os,ok,
                    ov.K_GRID,nd,ms,ov.N_INIT,
                    s+7919)
                rec[c]=dict(p=float(p),nm=nm,ns=nsd)
            out[sch]=rec
        return out
    except Exception:
        return{"error":traceback.format_exc()}


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        schemes=SCHEMES,constructions=list(CONSTRUCTIONS))
    s=ov.load_state(OUT,"E8",cfg)
    if s["cells"].get("done"):
        print("E8 already complete")
        return
    t0=time.time()
    tau,pt,ok,no=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E8] tau=%.4f tuned sphere-PR=%.3f ok=%s"%(tau,pt,ok),flush=True)
    ar=[(N,D,lam,i,400_000+1000*i,N_DRAWS)for i in range(NDS)]
    with Pool(47)as po:
        res=list(po.imap_unordered(worker,ar,chunksize=1))
    g=[r for r in res if "error" not in r]
    er=[r["error"]for r in res if "error" in r]
    a=ov.ALPHA
    ce={}
    for sch in SCHEMES:
        ro=[r[sch]for r in g if r.get(sch)]
        if not ro:
            ce[sch]=dict(scheme=sch,failed=True)
            continue
        e=dict(scheme=sch,n_datasets=len(ro),n_draws=N_DRAWS,
            pr_mean=float(np.mean([x["pr"]for x in ro])),
            pr_sd=float(np.std([x["pr"]for x in ro],ddof=1)),
            obs_mean=float(np.mean([x["obs"]for x in ro])),
            k_counts={int(k):int(sum(1 for x in ro if x["k"]==k))
            for k in sorted({x["k"]for x in ro})},
            failed=False)
        ps={}
        for cn in CONSTRUCTIONS:
            p=np.array([x[cn]["p"]for x in ro])
            ps[cn]=p
            cnt,Nn=int((p<a).sum()),p.size
            e[cn]=dict(rej=cnt/Nn,count=cnt,
                ci=list(nc._wilson(cnt,Nn)),
                mean_p=float(p.mean()),
                ks=float(nc._ks_uniform(p)),
                null_mean=float(np.mean([x[cn]["nm"]for x in ro])),
                null_sd=float(np.mean([x[cn]["ns"]for x in ro])))
        e["M_only"]=int(((ps["M"]<a)&(ps["C"]>=a)).sum())
        e["C_only"]=int(((ps["C"]<a)&(ps["M"]>=a)).sum())
        ce[sch]=e
    s["cells"]=ce
    s["meta"]=dict(tau=float(tau),tuned_sphere_pr=pt,tuned_ok=ok,
        n_errors=len(er),wall_s=round(time.time()-t0,1),
        error_sample=er[0]if er else None)
    ov.save_state(OUT,s)
    for sch,c in ce.items():
        if c.get("failed"):
            print("[E8] %-11s FAILED"%sch,flush=True)
            continue
        print("[E8] %-11s PR=%7.2f  M=%.3f(%d) G=%.3f(%d) S=%.3f(%d) C=%.3f(%d) "
            "Monly=%d Conly=%d"
            %(sch,c["pr_mean"],c["M"]["rej"],c["M"]["count"],
            c["G"]["rej"],c["G"]["count"],c["S"]["rej"],c["S"]["count"],
            c["C"]["rej"],c["C"]["count"],c["M_only"],c["C_only"]),
            flush=True)
    print("E8 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
