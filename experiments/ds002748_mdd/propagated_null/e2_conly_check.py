import json,time,traceback
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
E2=ov.OUTDIR/"e2_spectrum_shape.json"
OUT=ov.OUTDIR/"e2_conly_check.json"
N=28
D=ov.D
NDS=400
N_DRAWS=200
DEEP_DRAWS=2000
EXTRA_BASES=[810_000,820_000,830_000]


def rebuild_lam(cell):
    fam,pv=cell["family"],cell["param_value"]
    if fam=="exp":
        return ov.spec_exp(D,float(pv))
    if fam=="power":
        return ov.spec_power(D,float(pv))
    if fam=="one_spike":
        return ov.spec_one_spike(D,float(pv))
    if fam=="few_spikes":
        return ov.spec_few_spikes(D,float(pv))
    if fam=="two_block":
        m,eps=str(pv).split("/")
        return ov.spec_two_block(D,int(m),float(eps))
    raise ValueError(fam)


def worker(args):
    n,d,lam,idx,s,nd=args
    try:
        Xn=ov.gen(n,d,lam,s)
        ms=nc.min_cluster_guard(n)
        obs=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=ms,
            n_init=ov.N_INIT,random_state=idx)
        if obs[1]is None:
            return{"skip":True}
        kw=dict(k_grid=ov.K_GRID,n_draws=nd,min_size=ms,n_init=ov.N_INIT,
            seed=s+7919,observed=(obs[0],obs[1]))
        rM=nc.null_test(Xn,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,**kw)
        return dict(idx=idx,seed=s,pM=float(rM.p),pC=float(rC.p),
            obs=float(obs[0]),k=int(obs[1]),
            exM=int(rM.n_exceed),exC=int(rC.n_exceed),
            nmM=float(rM.null_mean),nmC=float(rC.null_mean))
    except Exception:
        return{"error":traceback.format_exc()}


def run_rep(pool,lam,base,off,n_draws,nds):
    ar=[(N,D,lam,i,base+1000*i+off,n_draws)for i in range(nds)]
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
    if not E2.exists():
        print("e2_spectrum_shape.json not found")
        return
    e2=json.loads(E2.read_text())
    fl={k:c for k,c in e2["cells"].items()
        if not c.get("failed")and c.get("C_only",0)>0}
    print("flagged cells: %s"%(list(fl)or "none"),flush=True)
    s=ov.load_state(OUT,"E2_conly_check",
        dict(n_datasets=NDS,n_draws=N_DRAWS,
        deep_draws=DEEP_DRAWS,extra_bases=EXTRA_BASES))
    if not fl:
        s["cells"]["none_flagged"]=True
        ov.save_state(OUT,s)
        return
    t0=time.time()
    f=["exp","power","one_spike","few_spikes","two_block"]
    with Pool(47)as po:
        for key,c in fl.items():
            if key in s["cells"]:
                print("  skip %s"%key,flush=True)
                continue
            tc=time.time()
            lam=rebuild_lam(c)
            off=int(round(c["pr_target"]*10))*100+f.index(c["family"])
            re,di=[],[]
            r0,d0=run_rep(po,lam,300_000,off,N_DRAWS,NDS)
            r0["replication"]=0
            r0["note"]="reproduces the original E2 cell"
            re.append(r0)
            di+=[dict(replication=0,**g)for g in d0]
            print("[chk2] %-18s rep0 M=%.3f(%d) C=%.3f(%d) Monly=%d Conly=%d"
                %(key,r0["rej_M"],r0["count_M"],r0["rej_C"],r0["count_C"],
                r0["M_only"],r0["C_only"]),flush=True)
            for j,b in enumerate(EXTRA_BASES,start=1):
                rj,dj=run_rep(po,lam,b,off,N_DRAWS,NDS)
                rj["replication"]=j
                re.append(rj)
                di+=[dict(replication=j,**g)for g in dj]
                print("[chk2] %-18s rep%d M=%.3f(%d) C=%.3f(%d) Monly=%d Conly=%d"
                    %(key,j,rj["rej_M"],rj["count_M"],rj["rej_C"],
                    rj["count_C"],rj["M_only"],rj["C_only"]),flush=True)
            de=None
            if di:
                d=[(N,D,lam,g["idx"],g["seed"],DEEP_DRAWS)for g in di]
                dr=list(po.imap_unordered(worker,d,chunksize=1))
                bs={r["seed"]:r for r in dr if "pM" in r}
                st=0
                for g in di:
                    r=bs.get(g["seed"])
                    if r is None:
                        g["deep"]=None
                        continue
                    p=bool(r["pC"]<ov.ALPHA<=r["pM"])
                    g["deep"]=dict(pM=r["pM"],pC=r["pC"],persists=p)
                    st+=int(p)
                de=dict(tested=len(di),persisting=st,draws=DEEP_DRAWS,
                    persist_frac=st/max(len(di),1))
                print("[chk2] %-18s deep: persists at %d draws in %d of %d (%.1f%%)"
                    %(key,DEEP_DRAWS,st,len(di),
                    100.0*st/max(len(di),1)),flush=True)
            toc=sum(r["C_only"]for r in re)
            tn=sum(r["n_datasets"]for r in re)
            s["cells"][key]=dict(
                cell=key,family=c["family"],pr_target=c["pr_target"],
                param_name=c["param_name"],param_value=c["param_value"],
                pr_mean_e2=c["pr_mean"],replications=re,
                total_C_only=toc,total_datasets=tn,
                C_only_rate=toc/tn,
                C_only_ci=list(nc._wilson(toc,tn)),
                n_discordant_recorded=len(di),
                discordant=di[:200],deep=de,
                wall_s=round(time.time()-tc,1))
            ov.save_state(OUT,s)
            print("[chk2] %-18s TOTAL C_only=%d / %d (rate %.5f)"
                %(key,toc,tn,toc/tn),flush=True)
    print("E2 verification done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
