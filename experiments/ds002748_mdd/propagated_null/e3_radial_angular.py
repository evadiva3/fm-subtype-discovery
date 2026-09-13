import json,time,traceback
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e3_radial_angular.json"
N=28
D=ov.D
PR_TARGET=4.4
NDS=400
N_DRAWS=200
FAMS=["N1","N2","N3","N4","N5","N6"]
RULES=["fixed","matched"]


def angular_factor(Xn):
    S=(Xn.T@Xn)/Xn.shape[0]
    w,V=np.linalg.eigh(S)
    w=np.clip(w,0.0,None)
    return np.sqrt(w)[:,None]*V.T


def draw(fam,Xn,mean,A,ang_A,n,d,rng):
    if fam=="N1":
        return mean+rng.standard_normal((n,A.shape[0]))@A
    if fam=="N2":
        return nc.l2_normalize(mean+rng.standard_normal((n,A.shape[0]))@A)
    if fam=="N3":
        U=nc.l2_normalize(mean+rng.standard_normal((n,A.shape[0]))@A)
        ro=np.linalg.norm(Xn,axis=1)
        r=rng.choice(ro,size=n,replace=True)[:,None]
        return U*r
    if fam=="N4":
        amb=mean+rng.standard_normal((n,A.shape[0]))@A
        r=np.linalg.norm(amb,axis=1,keepdims=True)
        return Xn*r
    if fam=="N5":
        ro=np.linalg.norm(Xn,axis=1)[:,None]
        return nc.l2_normalize(rng.standard_normal((n,d)))*ro
    if fam=="N6":
        return nc.l2_normalize(rng.standard_normal((n,d))@ang_A)
    raise ValueError(fam)


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
        aA=angular_factor(Xn)
        out={"pr":float(pr),"obs":float(os),"k":int(ok)}
        for fam in FAMS:
            rng=np.random.default_rng(se+7919+101*FAMS.index(fam))
            sf=np.empty(nd)
            ss=np.empty(nd)
            for b in range(nd):
                Dd=draw(fam,Xn,m,A,aA,n,d,rng)
                sf[b]=nc.silhouette_at_k(Dd,ok,n_init=ov.N_INIT,
                    random_state=b)
                sm,ks,_=nc.best_silhouette(Dd,k_grid=ov.K_GRID,min_size=ms,
                    n_init=ov.N_INIT,random_state=b)
                ss[b]=sm
            for r,s in(("fixed",sf),("matched",ss)):
                c=int((s>=os).sum())
                out["%s_%s"%(fam,r)]=dict(
                    p=float((c+1)/(nd+1)),
                    nm=float(s.mean()),ns=float(s.std(ddof=1)))
        return out
    except Exception:
        return{"error":traceback.format_exc()}


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        families=FAMS,rules=RULES)
    s=ov.load_state(OUT,"E3",cfg)
    if s["cells"].get("done"):
        print("E3 already complete")
        return
    t0=time.time()
    tau,pt,ok,no=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E3] tau=%.4f tuned PR=%.3f ok=%s %s"%(tau,pt,ok,no),flush=True)
    ar=[(N,D,lam,i,500_000+1000*i,N_DRAWS)for i in range(NDS)]
    with Pool(47)as po:
        res=list(po.imap_unordered(worker,ar,chunksize=1))
    g=[r for r in res if "obs" in r]
    e=[r["error"]for r in res if "error" in r]
    pr=np.array([r["pr"]for r in g])
    arm={}
    for fam in FAMS:
        for ru in RULES:
            kk="%s_%s"%(fam,ru)
            p=np.array([r[kk]["p"]for r in g])
            arm[kk]=ov.summarize_single(
                p,0.0,family=fam,rule=ru,
                null_mean=float(np.mean([r[kk]["nm"]for r in g])),
                null_sd=float(np.mean([r[kk]["ns"]for r in g])))
    pM=np.array([r["N1_fixed"]["p"]for r in g])
    pC=np.array([r["N2_matched"]["p"]for r in g])
    for kk,a in arm.items():
        p=np.array([r[kk]["p"]for r in g])
        a["only_vs_M"]=int(((p<ov.ALPHA)&(pM>=ov.ALPHA)).sum())
        a["M_only_vs_this"]=int(((pM<ov.ALPHA)&(p>=ov.ALPHA)).sum())
        a["only_vs_C"]=int(((p<ov.ALPHA)&(pC>=ov.ALPHA)).sum())
    s["cells"]=dict(
        done=True,tau=float(tau),tuned_pr=pt,tuned_ok=ok,tune_note=no,
        pr_mean=float(pr.mean()),pr_sd=float(pr.std(ddof=1)),
        n_datasets=len(g),n_errors=len(e),n_draws=N_DRAWS,
        obs_mean=float(np.mean([r["obs"]for r in g])),
        arms=arm,wall_s=round(time.time()-t0,1),
        error_sample=e[0]if e else None)
    ov.save_state(OUT,s)
    for kk,a in arm.items():
        print("[E3] %-12s rej=%.4f (%d/%d) [%.4f,%.4f] mean_p=%.4f KS=%.4f "
            "null_mean=%.4f sd=%.4f"
            %(kk,a["rej"],a["count"],a["n_datasets"],a["ci"][0],a["ci"][1],
            a["mean_p"],a["ks"],a["null_mean"],a["null_sd"]),flush=True)
    print("E3 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
