import json,time
from multiprocessing import Pool
import numpy as np
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e10_dimension.json"
CSV=ov.OUTDIR/"e10_dimension.csv"

PR_TARGET=4.4
DIMS=[32,64,128,256]
RATIOS=[0.24,1.0]
N_DRAWS=200
NDS_MAX=400
NDS_MIN=40
CELL_CAP_S=420.0
N_PROBE=3
CORES=47

CSV_COLS=["cell","d","n","n_over_d","pr_target","tau","pr_mean","pr_sd",
    "n_datasets","n_draws","rej_M","count_M","ci_M_lo","ci_M_hi",
    "rej_C","count_C","ci_C_lo","ci_C_hi","mean_pM","mean_pC",
    "ks_M","ks_C","M_only","C_only","wall_s"]


def main():
    cfg=dict(pr_target=PR_TARGET,dims=DIMS,ratios=RATIOS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        nds_max=NDS_MAX,nds_min=NDS_MIN,cell_cap_s=CELL_CAP_S)
    st=ov.load_state(OUT,"E10",cfg)
    t0=time.time()
    with Pool(CORES)as po:
        for d in DIMS:
            for r in RATIOS:
                n=max(8,int(round(r*d)))
                key="d%d_r%.2f"%(d,r)
                if key in st["cells"]:
                    print("  skip %s"%key,flush=True)
                    continue
                tc=time.time()
                tau,pt,ok,no=ov.bisect_param(
                    n,d,lambda p:ov.spec_exp(d,p),0.02,1e5,PR_TARGET)
                lam=ov.spec_exp(d,tau)
                tp=time.time()
                pr=[ov.mc_worker((n,d,lam,i,2_000_000+i,N_DRAWS,
                    ov.K_GRID))for i in range(N_PROBE)]
                pd=(time.time()-tp)/N_PROBE
                bd=int(CELL_CAP_S*CORES/max(pd,1e-6))
                nds=int(min(NDS_MAX,max(NDS_MIN,bd)))
                s=nds<NDS_MAX
                nb=sum(1 for p in pr if "pM" not in p)
                print("[E10] %-12s n=%-4d d=%-4d tau=%8.4f PR=%.2f ok=%s | probe "
                    "%.2f s/ds -> n_datasets=%d%s (probe_bad=%d)"
                    %(key,n,d,tau,pt,ok,pd,nds,
                    " SHRUNK" if s else "",nb),flush=True)
                a=[(n,d,lam,i,1_700_000+1000*i+d,N_DRAWS,
                    ov.K_GRID)for i in range(nds)]
                res=list(po.imap_unordered(ov.mc_worker,a,chunksize=1))
                c=ov.summarize_mc(
                    res,time.time()-tc,d=d,n=n,n_over_d=round(n/d,4),
                    pr_target=PR_TARGET,tau=float(tau),tuned_pr=pt,
                    tuned_ok=ok,tune_note=no,n_draws=N_DRAWS,
                    n_datasets_requested=nds,shrunk=s,
                    probe_s_per_dataset=round(pd,3))
                st["cells"][key]=c
                ov.save_state(OUT,st)
                ov.write_csv(CSV,st["cells"],CSV_COLS)
                if c.get("failed"):
                    print("[E10] %-12s FAILED errors=%d"%(key,c["n_errors"]),
                        flush=True)
                else:
                    f="  <<< C_only>0" if c["C_only"]>0 else ""
                    print("[E10] %-12s PR=%5.2f M=%.3f(%d) C=%.3f(%d) Monly=%d "
                        "Conly=%d [%.0fs tot %.0fs]%s"
                        %(key,c["pr_mean"],c["rej_M"],c["count_M"],
                        c["rej_C"],c["count_C"],c["M_only"],
                        c["C_only"],c["wall_s"],time.time()-t0,
                        f),flush=True)
    print("E10 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
