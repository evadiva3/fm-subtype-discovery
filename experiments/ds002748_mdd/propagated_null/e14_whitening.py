import json,time
import numpy as np
import ov_common as ov
import nullcal as nc
import e8_normalization as e8
OUT=ov.OUTDIR/"e14_whitening.json"
N=28
D=ov.D
PR_TARGET=4.4
N_CHECK=200

def pdist_matrix(X):
    A=X@X.T
    sq=np.diag(A)
    d2=np.maximum(sq[:,None]+sq[None,:]-2*A,0.0)
    return np.sqrt(d2)

def zca_truncated(X,rtol):
    X=np.asarray(X,dtype=np.float64)
    Xc=X-X.mean(axis=0,keepdims=True)
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    if S.size==0 or S[0]==0:
        return (Xc,0,np.inf)
    r=int((S>rtol*S[0]).sum())
    if r==0:
        return (Xc,0,np.inf)
    n=X.shape[0]
    W=Vt[:r].T* (np.sqrt(n-1)/S[:r])@Vt[:r]
    c=float(S[0]/S[r-1])
    return (Xc@W,r,c)

def main():
    t0=time.time()
    tau,pt,k,_=ov.bisect_param(N,D,lambda p:ov.spec_exp(D,p),0.02,100000.0,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E14] tau=%.4f tuned PR=%.3f"% (tau,pt),flush=True)
    ro=[]
    for rep in range(N_CHECK):
        rng=np.random.default_rng(400000+1000*rep)
        X=rng.standard_normal((N,D))@ov.factor(D,lam,rng)
        a=e8.t_whiten(X)
        b=e8.t_pca_whiten(X)
        c,dP=(pdist_matrix(a),pdist_matrix(b))
        sc=max(dP.max(),1e-300)
        rel=float(np.abs(c-dP).max()/sc)
        _,l,sv,m,rr=e8._svd_parts(X)
        d=float(sv[0]/sv[rr-1]) if rr>0 else np.inf
        co={}
        for rl in (1e-12,1e-10,1e-08,1e-06,0.0001):
            e,f,g=zca_truncated(X,rl)
            h=pdist_matrix(e)
            co["%g"%rl]=dict(rank=f,cond=g,rel_dev=float(np.abs(h-dP).max()/sc))
        ro.append(dict(rel_dev_default=rel,rank_default=int(rr),cond_default=d,truncated=co))
    rel=np.array([r["rel_dev_default"] for r in ro])
    ra=np.array([r["rank_default"] for r in ro])
    cond=np.array([r["cond_default"] for r in ro])
    su=dict(n_checked=N_CHECK,tau=float(tau),tuned_pr=pt,rel_dev_default=dict(mean=float(rel.mean()),median=float(np.median(rel)),max=float(rel.max()),min=float(rel.min())),rank_default=dict(mean=float(ra.mean()),min=int(ra.min()),max=int(ra.max())),cond_default=dict(median=float(np.median(cond)),max=float(cond.max())),truncated={})
    for rl in ("1e-12","1e-10","1e-08","1e-06","0.0001"):
        key=rl if rl in ro[0]["truncated"] else None
        if key is None:
            for t in ro[0]["truncated"]:
                if abs(float(t)-float(rl))<1e-30:
                    key=t
        if key is None:
            continue
        rd=np.array([r["truncated"][key]["rel_dev"] for r in ro])
        i=np.array([r["truncated"][key]["rank"] for r in ro])
        j=np.array([r["truncated"][key]["cond"] for r in ro])
        su["truncated"][key]=dict(rel_dev_median=float(np.median(rd)),rel_dev_max=float(rd.max()),rank_mean=float(i.mean()),cond_median=float(np.median(j)))
    ov.save_state(OUT, {"experiment":"E14","config":dict(n=N,d=D,pr_target=PR_TARGET,n_checked=N_CHECK),"cells":su})
    summ=su
    print("[E14] default rank %.1f (range %d-%d), condition number median %.3e"% (summ["rank_default"]["mean"],summ["rank_default"]["min"],summ["rank_default"]["max"],summ["cond_default"]["median"]),flush=True)
    print("[E14] ZCA vs PCA pairwise-distance relative deviation, default cutoff: median %.3e, max %.3e"% (summ["rel_dev_default"]["median"],summ["rel_dev_default"]["max"]),flush=True)
    print("[E14] with an explicit singular-value cutoff:",flush=True)
    for t,v in summ["truncated"].items():
        print("        rtol=%-8s rank %.1f  cond %.3e  rel dev median %.3e max %.3e"% (t,v["rank_mean"],v["cond_median"],v["rel_dev_median"],v["rel_dev_max"]),flush=True)
    print("done in %.0fs"% (time.time()-t0))
if __name__=="__main__":
    main()
