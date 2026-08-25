import json,time,traceback
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e13_mechanism.json"
N=28
D=ov.D
PR_TARGET=4.5
NDS=400
N_DRAWS=200
FAMS=["N1","N2","N3","N4","N5","N6"]
RULES=["fixed","matched"]

SPECTRA={
    "exp":dict(lo=0.02,hi=1e5,make=lambda d,p:ov.spec_exp(d,p),
    param="tau",reverses=False),
    "one_spike":dict(lo=1.0,hi=1e7,make=lambda d,p:ov.spec_one_spike(d,p),
    param="A",reverses=True),
    }


def angular_factor(Xn):
    S=(Xn.T@Xn)/Xn.shape[0]
    w,V=np.linalg.eigh(S)
    return np.sqrt(np.clip(w,0.0,None))[:,None]*V.T


def draw(fam,Xn,mean,A,ang_A,n,d,rng):
    if fam=="N1":
        return mean+rng.standard_normal((n,A.shape[0]))@A
    if fam=="N2":
        return nc.l2_normalize(mean+rng.standard_normal((n,A.shape[0]))@A)
    if fam=="N3":
        U=nc.l2_normalize(mean+rng.standard_normal((n,A.shape[0]))@A)
        r=rng.choice(np.linalg.norm(Xn,axis=1),size=n,replace=True)[:,None]
        return U*r
    if fam=="N4":
        amb=mean+rng.standard_normal((n,A.shape[0]))@A
        return Xn*np.linalg.norm(amb,axis=1,keepdims=True)
    if fam=="N5":
        r=np.linalg.norm(Xn,axis=1)[:,None]
        return nc.l2_normalize(rng.standard_normal((n,d)))*r
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
            return{"skip":True}
        m,A=nc._gaussian_factor(Xn)
        aA=angular_factor(Xn)
        out={"pr":float(pr),"obs":float(os),"k":int(ok)}
        for fam in FAMS:
            rng=np.random.default_rng(se+7919+101*FAMS.index(fam))
            sf=np.empty(nd)
            sm=np.empty(nd)
            for b in range(nd):
                Dd=draw(fam,Xn,m,A,aA,n,d,rng)
                sf[b]=nc.silhouette_at_k(Dd,ok,n_init=ov.N_INIT,random_state=b)
                sm[b]=nc.best_silhouette(Dd,k_grid=ov.K_GRID,min_size=ms,
                    n_init=ov.N_INIT,random_state=b)[0]
            for r,s in(("fixed",sf),("matched",sm)):
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
        families=FAMS,rules=RULES,spectra=list(SPECTRA)+["two_block"])
    st=ov.load_state(OUT,"E13",cfg)
    t0=time.time()
    with Pool(47)as po:
        for na in list(SPECTRA)+["two_block"]:
            if na in st["cells"]:
                print("  skip %s"%na,flush=True)
                continue
            tc=time.time()
            if na=="two_block":
                par,pt,ok,no=ov.tune_two_block(N,D,PR_TARGET)
                lam=ov.spec_two_block(D,par["m"],par["eps"])
                pd="m=%d eps=%.6g"%(par["m"],par["eps"])
            else:
                s=SPECTRA[na]
                par,pt,ok,no=ov.bisect_param(
                    N,D,lambda p,s=s:s["make"](D,p),s["lo"],s["hi"],PR_TARGET)
                lam=s["make"](D,par)
                pd="%s=%.6g"%(s["param"],par)
            print("[E13] %-11s %s tuned PR=%.3f ok=%s"%(na,pd,pt,ok),
                flush=True)
            off=(list(SPECTRA)+["two_block"]).index(na)
            ar=[(N,D,lam,i,1_900_000+1000*i+off,N_DRAWS)
                for i in range(NDS)]
            res=list(po.imap_unordered(worker,ar,chunksize=1))
            g=[r for r in res if "obs" in r]
            e=[r["error"]for r in res if "error" in r]
            a=ov.ALPHA
            arm={}
            for fam in FAMS:
                for ru in RULES:
                    kk="%s_%s"%(fam,ru)
                    p=np.array([r[kk]["p"]for r in g])
                    cnt=int((p<a).sum())
                    arm[kk]=dict(
                        family=fam,rule=ru,n_datasets=len(g),
                        rej=cnt/len(g),count=cnt,
                        ci=list(nc._wilson(cnt,len(g))),
                        mean_p=float(p.mean()),ks=float(nc._ks_uniform(p)),
                        null_mean=float(np.mean([r[kk]["nm"]for r in g])),
                        null_sd=float(np.mean([r[kk]["ns"]for r in g])))
            om=float(np.mean([r["obs"]for r in g]))
            sf=arm["N2_fixed"]["null_mean"]-arm["N1_fixed"]["null_mean"]
            sm=arm["N2_matched"]["null_mean"]-arm["N1_matched"]["null_mean"]
            st["cells"][na]=dict(
                spectrum=na,param=pd,tuned_pr=pt,tuned_ok=ok,
                pr_mean=float(np.mean([r["pr"]for r in g])),
                n_datasets=len(g),n_errors=len(e),n_draws=N_DRAWS,
                obs_mean=om,arms=arm,
                reprojection_null_mean_shift_fixed=sf,
                reprojection_null_mean_shift_matched=sm,
                M_rej=arm["N1_fixed"]["rej"],C_rej=arm["N2_matched"]["rej"],
                wall_s=round(time.time()-tc,1))
            ov.save_state(OUT,st)
            print("[E13] %-11s obs=%.4f | N1_fixed(M) mean=%.4f rej=%.3f | "
                "N2_matched(C) mean=%.4f rej=%.3f | reprojection shift "
                "fixed=%+.4f matched=%+.4f [%.0fs]"
                %(na,om,arm["N1_fixed"]["null_mean"],
                arm["N1_fixed"]["rej"],arm["N2_matched"]["null_mean"],
                arm["N2_matched"]["rej"],sf,sm,
                time.time()-tc),flush=True)
            for kk,x in arm.items():
                print("        %-12s mean=%.4f sd=%.4f rej=%.4f (%d/%d)"
                    %(kk,x["null_mean"],x["null_sd"],x["rej"],x["count"],
                    x["n_datasets"]),flush=True)
    print("E13 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
