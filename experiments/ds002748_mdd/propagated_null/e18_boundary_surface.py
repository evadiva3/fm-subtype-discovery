import json,time,traceback
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e18_boundary_surface.json"
CSV=ov.OUTDIR/"e18_boundary_surface.csv"
PR_TARGETS=[3.0,4.5,6.0]
NS=[28,60,119]
DS=[64,119,256]
SPIKES=[1.0,2.0,3.0,4.0,6.0,9.0,14.0]
NDS=150
N_DRAWS=100
TUNE_M=100
def spec_spike_exp(d,S,tau):
    lam=np.exp(-np.arange(d)/tau)
    lam[0]=float(S)
    return lam


def pc1_fraction(Xn):
    Xc=Xn-Xn.mean(axis=0,keepdims=True)
    s=np.linalg.svd(Xc,compute_uv=False)**2
    t=s.sum()
    return float(s[0]/t)if t>0 else float("nan")


def cell(args):
    pt,n,d,S,se=args
    try:
        t0=time.time()
        tau,prt,ok,no=ov.bisect_param(
            n,d,lambda t:spec_spike_exp(d,S,t),0.02,1e5,pt,m=TUNE_M)
        lam=spec_spike_exp(d,S,tau)
        ms=nc.min_cluster_guard(n)
        prs,p,m1s,m2s,s,pM,pC=[],[],[],[],[],[],[]
        for i in range(NDS):
            Xn=ov.gen(n,d,lam,se+1000*i)
            obs,obk,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=ms,
                n_init=ov.N_INIT,random_state=i)
            if obk is None:
                continue
            prs.append(nc.participation_ratio(Xn))
            p.append(pc1_fraction(Xn))
            me,A=nc._gaussian_factor(Xn)
            r1=np.random.default_rng(se+1000*i+7919)
            r2=np.random.default_rng(se+1000*i+7919)
            r3=np.random.default_rng(se+1000*i+7919)
            s1=np.empty(N_DRAWS);s2=np.empty(N_DRAWS);s3=np.empty(N_DRAWS)
            for b in range(N_DRAWS):
                s1[b]=nc.silhouette_at_k(nc._draw(me,A,n,r1),obk,
                    n_init=ov.N_INIT,random_state=b)
                s2[b]=nc.silhouette_at_k(
                    nc.l2_normalize(nc._draw(me,A,n,r2)),obk,
                    n_init=ov.N_INIT,random_state=b)
                s3[b]=nc.best_silhouette(
                    nc.l2_normalize(nc._draw(me,A,n,r3)),k_grid=ov.K_GRID,
                    min_size=ms,n_init=ov.N_INIT,random_state=b)[0]
            m1s.append(s1.mean());m2s.append(s2.mean());s.append(s1.std(ddof=1))
            pM.append((int((s1>=obs).sum())+1)/(N_DRAWS+1))
            pC.append((int((s3>=obs).sum())+1)/(N_DRAWS+1))
        if not m1s:
            return dict(pr_target=pt,n=n,d=d,spike_S=S,failed=True,
                reason="no dataset produced a guard-passing k")
        pM=np.array(pM);pC=np.array(pC);a=ov.ALPHA
        m1=float(np.mean(m1s));m2=float(np.mean(m2s));sd1=float(np.mean(s))
        return dict(pr_target=pt,n=n,d=d,spike_S=S,tau=float(tau),
            tuned_pr=prt,tuned_ok=bool(ok),tune_note=no,
            n_datasets=len(m1s),n_draws=N_DRAWS,
            pr_mean=float(np.mean(prs)),pc1_mean=float(np.mean(p)),
            null_mean_ambient=m1,null_mean_sphere=m2,null_sd_ambient=sd1,
            shift=m2-m1,
            shift_in_sd=float((m2-m1)/sd1)if sd1>0 else None,
            rej_M=float((pM<a).mean()),rej_C=float((pC<a).mean()),
            count_M=int((pM<a).sum()),count_C=int((pC<a).sum()),
            wall_s=round(time.time()-t0,1),failed=False)
    except Exception:
        return dict(pr_target=pt,n=n,d=d,spike_S=S,failed=True,
            reason=traceback.format_exc()[-400:])


def crossings(cells):
    out=[]
    k=sorted({(c["pr_target"],c["n"],c["d"])for c in cells if not c.get("failed")})
    for pt,n,d in k:
        g=sorted((c for c in cells if not c.get("failed")
            and(c["pr_target"],c["n"],c["d"])==(pt,n,d)),
            key=lambda c:c["spike_S"])
        rec=dict(pr_target=pt,n=n,d=d,n_cells=len(g),
            pr_mean=float(np.mean([c["pr_mean"]for c in g])),
            pr_span=[float(min(c["pr_mean"]for c in g)),
            float(max(c["pr_mean"]for c in g))],
            shift_first=g[0]["shift_in_sd"],shift_last=g[-1]["shift_in_sd"])
        cs=cp=None
        for i in range(len(g)-1):
            a,b=g[i]["shift"],g[i+1]["shift"]
            if a>0>=b:
                t=a/(a-b)
                cs=float(np.exp(np.log(g[i]["spike_S"])+
                    t*(np.log(g[i+1]["spike_S"])-np.log(g[i]["spike_S"]))))
                cp=float(g[i]["pc1_mean"]+t*(g[i+1]["pc1_mean"]-g[i]["pc1_mean"]))
                break
        rec["crossing_S"]=cs
        rec["crossing_pc1"]=cp
        rec["bracketed"]=cs is not None
        out.append(rec)
    return out

CSV_COLS=["cell","pr_target","n","d","spike_S","tau","pr_mean","pc1_mean",
    "null_mean_ambient","null_mean_sphere","shift","shift_in_sd",
    "n_datasets","n_draws","rej_M","rej_C","wall_s"]


def main():
    cfg=dict(pr_targets=PR_TARGETS,ns=NS,ds=DS,spikes=SPIKES,
        n_datasets=NDS,n_draws=N_DRAWS,tune_samples=TUNE_M,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA)
    st=ov.load_state(OUT,"E18",cfg)
    jo=[]
    for pt in PR_TARGETS:
        for n in NS:
            for d in DS:
                for S in SPIKES:
                    key="pr%s_n%d_d%d_S%g"%(pt,n,d,S)
                    if key in st["cells"]:
                        continue
                    s=(2_500_000+int(pt*10)*100000+n*1000
                        +d*10+int(S))
                    jo.append((key,(pt,n,d,S,s)))
    print("[E18] %d cells to run (%d already present)"
        %(len(jo),len(st["cells"])),flush=True)
    t0=time.time()
    with Pool(47)as p:
        for(key,_),res in zip(jo,p.imap(cell,[j[1]for j in jo],chunksize=1)):
            st["cells"][key]=res
            if res.get("failed"):
                print("[E18] %-26s FAILED %s"%(key,res.get("reason","")[:60]),
                    flush=True)
            else:
                print("[E18] %-26s PR=%5.2f PC1=%.3f shift=%+.3f sd  M=%.3f C=%.3f "
                    "[%.0fs]"%(key,res["pr_mean"],res["pc1_mean"],
                    res["shift_in_sd"],res["rej_M"],res["rej_C"],
                    res["wall_s"]),flush=True)
            ov.save_state(OUT,st)
    g=[c for c in st["cells"].values()if isinstance(c,dict)and not c.get("failed")]
    ov.write_csv(CSV,{k:v for k,v in st["cells"].items()
        if isinstance(v,dict)and not v.get("failed")},CSV_COLS)
    cr=crossings(g)
    st["crossings"]=cr
    ov.save_state(OUT,st)
    print("\n[E18] crossings by geometry")
    print("%-8s %5s %5s %10s %8s %14s"%("PR","n","d","crossing_S","PC1","PR span"))
    for r in sorted(cr,key=lambda r:(r["pr_target"],r["n"],r["d"])):
        print("%-8.1f %5d %5d %10s %8s %6.2f-%.2f"
            %(r["pr_target"],r["n"],r["d"],
            ("%.2f"%r["crossing_S"])if r["bracketed"]else "not brkt",
            ("%.3f"%r["crossing_pc1"])if r["bracketed"]else "-",
            r["pr_span"][0],r["pr_span"][1]))
    print("E18 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
