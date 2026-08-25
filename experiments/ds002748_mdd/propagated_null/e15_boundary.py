import json,time,traceback
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e15_boundary.json"
CSV=ov.OUTDIR/"e15_boundary.csv"
N=28
D=ov.D
PR_TARGET=4.5
NDS=400
N_DRAWS=200
TUNE_M=200
SPIKES=[1.0,1.5,2.0,3.0,4.0,6.0,9.0,14.0,22.0,40.0,80.0,160.0]


def spec_spike_exp(d,S,tau):
    lam=np.exp(-np.arange(d)/tau)
    lam[0]=float(S)
    return lam


def pc1_fraction(Xn):
    Xc=Xn-Xn.mean(axis=0,keepdims=True)
    s=np.linalg.svd(Xc,compute_uv=False)**2
    tot=s.sum()
    return float(s[0]/tot)if tot>0 else float("nan")


def worker(args):
    n,d,lam,idx,se,nd=args
    try:
        Xn=ov.gen(n,d,lam,se)
        pr=nc.participation_ratio(Xn)
        pc1=pc1_fraction(Xn)
        ms=nc.min_cluster_guard(n)
        obs,ok,_=nc.best_silhouette(Xn,k_grid=ov.K_GRID,min_size=ms,
            n_init=ov.N_INIT,random_state=idx)
        if ok is None:
            return{"skip":True}
        m,A=nc._gaussian_factor(Xn)
        r1=np.random.default_rng(se+7919)
        r2=np.random.default_rng(se+7919)
        r3=np.random.default_rng(se+7919)
        s1=np.empty(nd)
        s2=np.empty(nd)
        s3=np.empty(nd)
        for b in range(nd):
            s1[b]=nc.silhouette_at_k(nc._draw(m,A,n,r1),ok,
                n_init=ov.N_INIT,random_state=b)
            s2[b]=nc.silhouette_at_k(nc.l2_normalize(nc._draw(m,A,n,r2)),
                ok,n_init=ov.N_INIT,random_state=b)
            s3[b]=nc.best_silhouette(nc.l2_normalize(nc._draw(m,A,n,r3)),
                k_grid=ov.K_GRID,min_size=ms,
                n_init=ov.N_INIT,random_state=b)[0]
        p=lambda s:float((int((s>=obs).sum())+1)/(nd+1))
        return dict(pr=float(pr),pc1=pc1,obs=float(obs),k=int(ok),
            pM=p(s1),pC=p(s3),p2f=p(s2),
            m1=float(s1.mean()),m2=float(s2.mean()),m3=float(s3.mean()),
            sd1=float(s1.std(ddof=1)),sd2=float(s2.std(ddof=1)),
            sd3=float(s3.std(ddof=1)))
    except Exception:
        return{"error":traceback.format_exc()}

CSV_COLS=["cell","spike_S","tau","pr_mean","pc1_mean","obs_mean",
    "null_mean_ambient","null_mean_sphere","shift","shift_in_sd",
    "n_datasets","rej_M","count_M","ci_M_lo","ci_M_hi",
    "rej_C","count_C","ci_C_lo","ci_C_hi","M_only","C_only"]


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        spikes=SPIKES,
        family="lambda_0 = S, lambda_i = exp(-i/tau); tau bisected to hold PR")
    st=ov.load_state(OUT,"E15",cfg)
    t0=time.time()
    a=ov.ALPHA
    with Pool(47)as p:
        for S in SPIKES:
            key="S%g"%S
            if key in st["cells"]:
                print("  skip %s"%key,flush=True)
                continue
            tc=time.time()
            tau,pt,ok,no=ov.bisect_param(
                N,D,lambda t,S=S:spec_spike_exp(D,S,t),0.02,1e5,PR_TARGET,m=TUNE_M)
            lam=spec_spike_exp(D,S,tau)
            ar=[(N,D,lam,i,2_100_000+1000*i+int(S*7),N_DRAWS)
                for i in range(NDS)]
            res=list(p.imap_unordered(worker,ar,chunksize=1))
            g=[r for r in res if "pM" in r]
            if not g:
                st["cells"][key]=dict(spike_S=S,failed=True)
                ov.save_state(OUT,st)
                print("[E15] %-7s FAILED"%key,flush=True)
                continue
            pM=np.array([r["pM"]for r in g])
            pC=np.array([r["pC"]for r in g])
            m1=float(np.mean([r["m1"]for r in g]))
            m2=float(np.mean([r["m2"]for r in g]))
            sd1=float(np.mean([r["sd1"]for r in g]))
            cM,cC,Nn=int((pM<a).sum()),int((pC<a).sum()),len(g)
            s=m2-m1
            c=dict(
                spike_S=float(S),tau=float(tau),tuned_pr=pt,tuned_ok=ok,
                tune_note=no,n_datasets=Nn,n_draws=N_DRAWS,
                pr_mean=float(np.mean([r["pr"]for r in g])),
                pr_sd=float(np.std([r["pr"]for r in g],ddof=1)),
                pc1_mean=float(np.mean([r["pc1"]for r in g])),
                pc1_sd=float(np.std([r["pc1"]for r in g],ddof=1)),
                obs_mean=float(np.mean([r["obs"]for r in g])),
                null_mean_ambient=m1,null_mean_sphere=m2,
                null_sd_ambient=sd1,
                null_mean_sphere_matched=float(np.mean([r["m3"]for r in g])),
                shift=s,shift_in_sd=float(s/sd1)if sd1>0 else None,
                rej_M=cM/Nn,count_M=cM,ci_M=list(nc._wilson(cM,Nn)),
                rej_C=cC/Nn,count_C=cC,ci_C=list(nc._wilson(cC,Nn)),
                mean_pM=float(pM.mean()),mean_pC=float(pC.mean()),
                ks_M=float(nc._ks_uniform(pM)),ks_C=float(nc._ks_uniform(pC)),
                M_only=int(((pM<a)&(pC>=a)).sum()),
                C_only=int(((pC<a)&(pM>=a)).sum()),
                wall_s=round(time.time()-tc,1),failed=False)
            st["cells"][key]=c
            ov.save_state(OUT,st)
            ov.write_csv(CSV,st["cells"],CSV_COLS)
            print("[E15] S=%-6g tau=%8.3f PR=%5.2f PC1=%.3f | ambient=%.4f "
                "sphere=%.4f shift=%+.4f (%+.2f sd) | M=%.3f C=%.3f "
                "Monly=%d Conly=%d [%.0fs]"
                %(S,tau,c["pr_mean"],c["pc1_mean"],m1,m2,s,
                c["shift_in_sd"]or 0.0,c["rej_M"],c["rej_C"],
                c["M_only"],c["C_only"],c["wall_s"]),flush=True)
    print("E15 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
