import json,time,traceback
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
from e6_statistics import best_stat,stat_at_k,DEGENERATE
OUT=ov.OUTDIR/"e16_nwb_check.json"
N=28
D=ov.D
PR_TARGET=4.4
NDS=400
N_DRAWS=200
DEEP_DRAWS=2000
STATS=["nwb","ndb"]
BASES=[100_000,910_000,920_000,930_000]


def worker(args):
    n,d,lam,idx,se,nd,st=args
    try:
        Xn=ov.gen(n,d,lam,se)
        ms=nc.min_cluster_guard(n)
        obs,ok=best_stat(st,Xn,ov.K_GRID,ms,ov.N_INIT,idx)
        if ok is None:
            return{"skip":True}
        rM=np.random.default_rng(se+7919)
        rC=np.random.default_rng(se+7919)
        m,A=nc._gaussian_factor(Xn)
        sM=np.empty(nd)
        sC=np.empty(nd)
        for b in range(nd):
            sM[b]=stat_at_k(st,nc._draw(m,A,n,rM),ok,ov.N_INIT,b)
            sC[b]=best_stat(st,nc.l2_normalize(nc._draw(m,A,n,rC)),
                ov.K_GRID,ms,ov.N_INIT,b)[0]
        f=lambda s:float((int((s>=obs).sum())+1)/(nd+1))
        return dict(idx=idx,seed=se,stat=st,pM=f(sM),pC=f(sC),
            obs=float(obs),k=int(ok))
    except Exception:
        return{"error":traceback.format_exc()}


def run_rep(pool,lam,base,stat,n_draws,nds):
    ar=[(N,D,lam,i,base+1000*i,n_draws,stat)for i in range(nds)]
    res=list(pool.imap_unordered(worker,ar,chunksize=1))
    g=[r for r in res if "pM" in r]
    pM=np.array([r["pM"]for r in g])
    pC=np.array([r["pC"]for r in g])
    a=ov.ALPHA
    m=(pC<a)&(pM>=a)
    cM,cC,Nn=int((pM<a).sum()),int((pC<a).sum()),len(g)
    rec=dict(base=base,n_datasets=Nn,
        rej_M=cM/Nn,count_M=cM,ci_M=list(nc._wilson(cM,Nn)),
        rej_C=cC/Nn,count_C=cC,ci_C=list(nc._wilson(cC,Nn)),
        M_only=int(((pM<a)&(pC>=a)).sum()),C_only=int(m.sum()))
    return rec,[g[int(j)]for j in np.flatnonzero(m)]


def main():
    st=ov.load_state(OUT,"E16",dict(
        n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        deep_draws=DEEP_DRAWS,statistics=STATS,seed_bases=BASES))
    t0=time.time()
    tau,pt,ok,_=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E16] tau=%.4f tuned PR=%.3f"%(tau,pt),flush=True)
    with Pool(47)as po:
        for s in STATS:
            if s in st["cells"]:
                print("  skip %s"%s,flush=True)
                continue
            tc=time.time()
            re,di=[],[]
            for j,b in enumerate(BASES):
                r,d0=run_rep(po,lam,b,s,N_DRAWS,NDS)
                r["replication"]=j
                if j==0:
                    r["note"]="reproduces the E6 seeding"
                re.append(r)
                di+=[dict(replication=j,**g)for g in d0]
                print("[E16] %-4s rep%d M=%.4f(%d) C=%.4f(%d) Monly=%d Conly=%d"
                    %(s,j,r["rej_M"],r["count_M"],r["rej_C"],
                    r["count_C"],r["M_only"],r["C_only"]),flush=True)
            de=None
            if di:
                da=[(N,D,lam,g["idx"],g["seed"],DEEP_DRAWS,s)
                    for g in di]
                dr=list(po.imap_unordered(worker,da,chunksize=1))
                by={r["seed"]:r for r in dr if "pM" in r}
                sl=0
                for g in di:
                    r=by.get(g["seed"])
                    if r is None:
                        g["deep"]=None
                        continue
                    pe=bool(r["pC"]<ov.ALPHA<=r["pM"])
                    g["deep"]=dict(pM=r["pM"],pC=r["pC"],persists=pe)
                    sl+=int(pe)
                de=dict(tested=len(di),persisting=sl,draws=DEEP_DRAWS,
                    persist_frac=sl/max(len(di),1))
                print("[E16] %-4s deep: persists at %d draws in %d of %d (%.1f%%)"
                    %(s,DEEP_DRAWS,sl,len(di),
                    100.0*sl/max(len(di),1)),flush=True)
            toc=sum(r["C_only"]for r in re)
            tm=sum(r["M_only"]for r in re)
            tn=sum(r["n_datasets"]for r in re)
            st["cells"][s]=dict(
                statistic=s,tau=float(tau),replications=re,
                total_C_only=toc,total_M_only=tm,total_datasets=tn,
                C_only_rate=toc/tn,
                C_only_ci=list(nc._wilson(toc,tn)),
                pooled_M=float(sum(r["count_M"]for r in re)/tn),
                pooled_C=float(sum(r["count_C"]for r in re)/tn),
                pooled_M_ci=list(nc._wilson(sum(r["count_M"]for r in re),tn)),
                pooled_C_ci=list(nc._wilson(sum(r["count_C"]for r in re),tn)),
                discordant=di[:200],deep=de,
                wall_s=round(time.time()-tc,1))
            ov.save_state(OUT,st)
            c=st["cells"][s]
            print("[E16] %-4s POOLED M=%.4f %s  C=%.4f %s  C_only=%d M_only=%d /%d"
                %(s,c["pooled_M"],[round(x,4)for x in c["pooled_M_ci"]],
                c["pooled_C"],[round(x,4)for x in c["pooled_C_ci"]],
                toc,tm,tn),flush=True)
    print("E16 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
