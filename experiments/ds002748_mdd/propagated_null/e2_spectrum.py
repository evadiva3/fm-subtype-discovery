import json,time
from multiprocessing import Pool
import numpy as np
import ov_common as ov
OUT=ov.OUTDIR/"e2_spectrum_shape.json"
CSV=ov.OUTDIR/"e2_spectrum_shape.csv"
N=28
D=ov.D
TARGETS=[3.0,4.5,6.0]
NDS=400
N_DRAWS=200

FAMILIES={
    "exp":dict(lo=0.02,hi=1e5,make=lambda d,p:ov.spec_exp(d,p),
    param="tau"),
    "power":dict(lo=0.001,hi=30.0,make=lambda d,p:ov.spec_power(d,p),
    param="a"),
    "one_spike":dict(lo=1.0,hi=1e7,make=lambda d,p:ov.spec_one_spike(d,p),
    param="A"),
    "few_spikes":dict(lo=1.0,hi=1e7,make=lambda d,p:ov.spec_few_spikes(d,p),
    param="A"),
    }

CSV_COLS=["cell","family","pr_target","param_name","param_value",
    "pr_mean","pr_sd","n_datasets","n_draws",
    "rej_M","count_M","ci_M_lo","ci_M_hi",
    "rej_C","count_C","ci_C_lo","ci_C_hi",
    "mean_pM","mean_pC","ks_M","ks_C",
    "null_mean_M","null_sd_M","null_mean_C","null_sd_C",
    "M_only","C_only","tuned_ok","wall_s"]


def main():
    cfg=dict(n=N,d=D,targets=TARGETS,n_datasets=NDS,n_draws=N_DRAWS,
        n_init=ov.N_INIT,k_grid=list(ov.K_GRID),alpha=ov.ALPHA,
        families=list(FAMILIES)+["two_block"],pr_tolerance=ov.TOL)
    s=ov.load_state(OUT,"E2",cfg)
    t0=time.time()
    with Pool(47)as po:
        for fam in list(FAMILIES)+["two_block"]:
            for tgt in TARGETS:
                key="%s_pr%s"%(fam,tgt)
                if key in s["cells"]:
                    print("  skip %s"%key,flush=True)
                    continue
                tc=time.time()
                if fam=="two_block":
                    par,pt,ok,no=ov.tune_two_block(N,D,tgt)
                    lam=ov.spec_two_block(D,par["m"],par["eps"])
                    pn,pv="m/eps","%d/%.6g"%(par["m"],par["eps"])
                else:
                    f=FAMILIES[fam]
                    par,pt,ok,no=ov.bisect_param(
                        N,D,lambda p,f=f:f["make"](D,p),f["lo"],f["hi"],tgt)
                    lam=f["make"](D,par)
                    pn,pv=f["param"],float(par)
                off=(int(round(tgt*10))*100
                    +(list(FAMILIES)+["two_block"]).index(fam))
                a=[(N,D,lam,i,300_000+1000*i+off,N_DRAWS,ov.K_GRID)
                    for i in range(NDS)]
                res=list(po.imap_unordered(ov.mc_worker,a,chunksize=1))
                c=ov.summarize_mc(res,time.time()-tc,family=fam,
                    pr_target=tgt,param_name=pn,
                    param_value=pv,tuned_pr=pt,
                    tuned_ok=ok,tune_note=no,n_draws=N_DRAWS,
                    n=N,d=D)
                s["cells"][key]=c
                ov.save_state(OUT,s)
                ov.write_csv(CSV,s["cells"],CSV_COLS)
                if c.get("failed"):
                    print("[E2] %-18s FAILED errors=%d"%(key,c["n_errors"]),
                        flush=True)
                else:
                    fl="  <<< C_only>0" if c["C_only"]>0 else ""
                    print("[E2] %-18s %s=%-12s PR=%5.2f(t%s%s) M=%.3f(%d) C=%.3f(%d) "
                        "Monly=%d Conly=%d [%.0fs tot %.0fs]%s"
                        %(key,pn,str(pv)[:12],c["pr_mean"],tgt,
                        "" if ok else "!",c["rej_M"],c["count_M"],
                        c["rej_C"],c["count_C"],c["M_only"],
                        c["C_only"],c["wall_s"],time.time()-t0,fl),
                        flush=True)
    print("E2 done in %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()
