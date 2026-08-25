import json,time,traceback
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e5_null_families.json"
N=28
D=ov.D
PR_TARGET=4.4
NDS=400
N_DRAWS=200
ARMS=["gaussian_reproj","isotropic","vmf","angular_gaussian",
    "coord_resample","random_rotation"]


def fit_vmf(U):
    d=U.shape[1]
    m=U.mean(axis=0)
    R=float(np.linalg.norm(m))
    if R<1e-12:
        return np.eye(d)[0],0.0
    mu=m/R
    R=min(R,1-1e-10)
    k=R*(d-R**2)/(1-R**2)
    return mu,float(max(k,0.0))


def rvmf(mu,kappa,n,rng):
    d=mu.size
    if kappa<1e-8:
        return nc.l2_normalize(rng.standard_normal((n,d)))
    b=(-2*kappa+np.sqrt(4*kappa**2+(d-1)**2))/(d-1)
    x0=(1-b)/(1+b)
    c=kappa*x0+(d-1)*np.log(max(1-x0**2,1e-300))
    out=np.empty((n,d))
    for i in range(n):
        for _ in range(10000):
            Z=rng.beta((d-1)/2.0,(d-1)/2.0)
            W=(1-(1+b)*Z)/(1-(1-b)*Z)
            U=rng.random()
            if kappa*W+(d-1)*np.log(max(1-x0*W,1e-300))-c>=np.log(U):
                break
        v=rng.standard_normal(d)
        v-=v.dot(mu)*mu
        nv=np.linalg.norm(v)
        v=v/nv if nv>0 else v
        out[i]=np.sqrt(max(1-W**2,0.0))*v+W*mu
    return out


def make_draw(arm,Xn,mean,A,ang_A,vmf_mu,vmf_k,n,d,rng):
    if arm=="gaussian_reproj":
        return nc.l2_normalize(nc._draw(mean,A,n,rng))
    if arm=="isotropic":
        return nc.l2_normalize(rng.standard_normal((n,d)))
    if arm=="vmf":
        return rvmf(vmf_mu,vmf_k,n,rng)
    if arm=="angular_gaussian":
        return nc.l2_normalize(rng.standard_normal((n,d))@ang_A)
    if arm=="coord_resample":
        idx=rng.integers(0,n,size=(n,d))
        return nc.l2_normalize(Xn[idx,np.arange(d)[None,:]])
    if arm=="random_rotation":
        Q,_=np.linalg.qr(rng.standard_normal((d,d)))
        return nc.l2_normalize(Xn@Q)
    raise ValueError(arm)


def worker(args):
    n,d,lam,idx,se,nd=args
    try:
        Xn=ov.gen(n,d,lam,se)
        pr=nc.participation_ratio(Xn)
        ms=nc.min_cluster_guard(n)
        os,ok,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=ms,
            n_init=ov.N_INIT,random_state=idx)
        if ok is None:
            return{"skip":"no k passes guard on observed"}
        m,A=nc._gaussian_factor(Xn)
        S=(Xn.T@Xn)/n
        w,V=np.linalg.eigh(S)
        aA=np.sqrt(np.clip(w,0,None))[:,None]*V.T
        vm,vk=fit_vmf(Xn)
        out={"pr":float(pr),"obs":float(os),"k":int(ok),
            "vmf_kappa":vk}
        for arm in ARMS:
            rng=np.random.default_rng(se+7919+137*ARMS.index(arm))
            s=np.empty(nd)
            for b in range(nd):
                Dd=make_draw(arm,Xn,m,A,aA,vm,vk,n,d,rng)
                sm,_,_=nc.best_silhouette(Dd,k_grid=ov.K_GRID,min_size=ms,
                    n_init=ov.N_INIT,random_state=b)
                s[b]=sm
            c=int((s>=os).sum())
            out[arm]=dict(p=float((c+1)/(nd+1)),
                nm=float(s.mean()),ns=float(s.std(ddof=1)))
        return out
    except Exception:
        return{"error":traceback.format_exc()}


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,arms=ARMS)
    s=ov.load_state(OUT,"E5",cfg)
    if s["cells"].get("done"):
        print("E5 already complete")
        return
    t0=time.time()
    tau,pt,ok,no=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E5] tau=%.4f tuned PR=%.3f ok=%s"%(tau,pt,ok),flush=True)
    ar=[(N,D,lam,i,800_000+1000*i,N_DRAWS)for i in range(NDS)]
    with Pool(47)as po:
        res=list(po.imap_unordered(worker,ar,chunksize=1))
    g=[r for r in res if "obs" in r]
    e=[r["error"]for r in res if "error" in r]
    pr=np.array([r["pr"]for r in g])
    pb=np.array([r["gaussian_reproj"]["p"]for r in g])
    a2={}
    for arm in ARMS:
        p=np.array([r[arm]["p"]for r in g])
        a2[arm]=ov.summarize_single(
            p,0.0,arm=arm,
            null_mean=float(np.mean([r[arm]["nm"]for r in g])),
            null_sd=float(np.mean([r[arm]["ns"]for r in g])),
            only_vs_C=int(((p<ov.ALPHA)&(pb>=ov.ALPHA)).sum()),
            C_only_vs_this=int(((pb<ov.ALPHA)&(p>=ov.ALPHA)).sum()))
    s["cells"]=dict(
        done=True,tau=float(tau),tuned_pr=pt,tuned_ok=ok,tune_note=no,
        pr_mean=float(pr.mean()),pr_sd=float(pr.std(ddof=1)),
        n_datasets=len(g),n_errors=len(e),n_draws=N_DRAWS,
        obs_mean=float(np.mean([r["obs"]for r in g])),
        vmf_kappa_mean=float(np.mean([r["vmf_kappa"]for r in g])),
        arms=a2,wall_s=round(time.time()-t0,1),
        error_sample=e[0]if e else None)
    ov.save_state(OUT,s)
    for arm,a in a2.items():
        print("[E5] %-17s rej=%.4f (%d/%d) [%.4f,%.4f] mean_p=%.4f KS=%.4f "
            "null_mean=%.4f"%(arm,a["rej"],a["count"],a["n_datasets"],
            a["ci"][0],a["ci"][1],a["mean_p"],a["ks"],
            a["null_mean"]),flush=True)
    print("E5 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
