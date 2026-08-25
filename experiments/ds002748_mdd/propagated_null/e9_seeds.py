import json,time
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e9_calibration_seeds.json"
N=28
D=ov.D
PR_TARGET=4.4
NDS=300
N_DRAWS=300
BASES=[1_100_000,1_200_000,1_300_000,1_400_000,1_500_000]


def main():
    cfg=dict(n=N,d=D,pr_target=PR_TARGET,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        n_replications=len(BASES),seed_bases=BASES)
    st=ov.load_state(OUT,"E9",cfg)
    t0=time.time()
    tau,pt,ok,no=ov.bisect_param(
        N,D,lambda p:ov.spec_exp(D,p),0.02,1e5,PR_TARGET)
    lam=ov.spec_exp(D,tau)
    print("[E9] tau=%.4f tuned PR=%.3f ok=%s"%(tau,pt,ok),flush=True)
    with Pool(47)as po:
        for rep,b in enumerate(BASES):
            key="rep%d"%rep
            if key in st["cells"]:
                print("  skip %s"%key,flush=True)
                continue
            tc=time.time()
            a=[(N,D,lam,i,b+1000*i,N_DRAWS,ov.K_GRID)
                for i in range(NDS)]
            res=list(po.imap_unordered(ov.mc_worker,a,chunksize=1))
            ce=ov.summarize_mc(res,time.time()-tc,replication=rep,
                seed_base=b,tau=float(tau),n=N,d=D,
                n_draws=N_DRAWS)
            st["cells"][key]=ce
            ov.save_state(OUT,st)
            f="  <<< C_only>0" if ce["C_only"]>0 else ""
            print("[E9] rep%d PR=%.2f M=%.4f(%d) [%.4f,%.4f]  C=%.4f(%d) "
                "[%.4f,%.4f]  Monly=%d Conly=%d [%.0fs]%s"
                %(rep,ce["pr_mean"],ce["rej_M"],ce["count_M"],
                ce["ci_M"][0],ce["ci_M"][1],ce["rej_C"],
                ce["count_C"],ce["ci_C"][0],ce["ci_C"][1],
                ce["M_only"],ce["C_only"],ce["wall_s"],f),
                flush=True)
    r=[st["cells"][k]for k in sorted(st["cells"])if k.startswith("rep")]
    if r:
        rM=np.array([c["rej_M"]for c in r])
        rC=np.array([c["rej_C"]for c in r])
        st["summary"]=dict(
            n_replications=len(r),
            M_rates=rM.tolist(),C_rates=rC.tolist(),
            M_mean=float(rM.mean()),M_sd=float(rM.std(ddof=1)),
            M_min=float(rM.min()),M_max=float(rM.max()),
            C_mean=float(rC.mean()),C_sd=float(rC.std(ddof=1)),
            C_min=float(rC.min()),C_max=float(rC.max()),
            C_counts=[c["count_C"]for c in r],
            M_counts=[c["count_M"]for c in r],
            M_only=[c["M_only"]for c in r],
            C_only=[c["C_only"]for c in r],
            C_max_over_min=(float(rC.max()/rC.min())if rC.min()>0 else None),
            pooled_C=list(nc._wilson(int(sum(c["count_C"]for c in r)),
            int(sum(c["n_datasets"]for c in r)))),
            pooled_M=list(nc._wilson(int(sum(c["count_M"]for c in r)),
            int(sum(c["n_datasets"]for c in r)))))
        ov.save_state(OUT,st)
        s=st["summary"]
        print("[E9] M rates %s  (mean %.4f sd %.4f)"
            %(["%.4f"%x for x in s["M_rates"]],s["M_mean"],s["M_sd"]),
            flush=True)
        print("[E9] C rates %s  (mean %.4f sd %.4f, counts %s)"
            %(["%.4f"%x for x in s["C_rates"]],s["C_mean"],s["C_sd"],
            s["C_counts"]),flush=True)
    print("E9 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
