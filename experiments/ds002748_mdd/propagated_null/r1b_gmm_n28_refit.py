#Phase 1B supplement: the two-component mixture arm at n = 28.
import json
import os
import platform
import sys
import time
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nullcal as nc
import ov_common as ov
import r1b_nongaussian as r1b
D=119;N=28;PR_TARGETS=[3.0,4.5,9.0];R_TARGET=0.6;NDS=400;SEED_ROOT=4250000;_CFG=None
def _init(cfg):
    global _CFG;r1b._CFG=cfg
def measure_floor(n,d,A,seed=987654,m=200):
    v=[]
    for j in range(m):
        rg=np.random.default_rng(seed+31*j);r,l=r1b.draw_raw("gmm_R0.3",n,d,A,0.0,rg);v.append(nc.separation_ratio(nc.l2_normalize(r),l))
    return(float(np.mean(v)),float(np.std(v,ddof=1)))
def main():
    s=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");o1=ov.ROOT/"results"/"phase1"/s;o1.mkdir(parents=True,exist_ok=True);o=o1/"r1b_gmm_n28_refit.json";t=time.time();cg,g=({},{})
    for pr in PR_TARGETS:
        k=f"pr{pr:g}_n{N}";tu,p,ok,n=ov.bisect_param(N,D,lambda t:ov.spec_exp(D,t),0.2,200.0,pr,tol=0.05,m=100);rg=np.random.default_rng(SEED_ROOT+hash(k)%100000);A=ov.factor(D,ov.spec_exp(D,tu),rg)
        f,fd=measure_floor(N,D,A);d,r=r1b.tune_delta(N,D,A,R_TARGET);cg[k]={"d":D,"A":A,"delta":{"gmm_R0.3":d}}
        g[k]={"pr_target":pr,"n":N,"tau":float(tu),"tuned_pr_gaussian":float(p),"tune_ok":bool(ok),"separation_ratio_floor_mean":f,"separation_ratio_floor_sd":fd,"r_target":R_TARGET,"gmm_delta":float(d),"gmm_separation_ratio_measured":float(r)}
        print(f"[tune] {k:12s} floor R={f:.4f} (sd {fd:.4f})  delta={d:.4f} -> R={r:.4f}",flush=True)
    import scipy
    import sklearn
    se={"header":{"experiment":"R1B mixture arm refit at n = 28","purpose":"the R = 0.3 target is below the separation ratio floor at n = 28; rerun at 0.6","written_utc":s,"d":D,"n":N,"pr_targets":PR_TARGETS,"r_target":R_TARGET,"r_target_rationale":"reachable at n = 28, and half the article's detection floor of 1.17","n_datasets_per_cell":NDS,"n_draws":r1b.N_DRAWS,"n_init":r1b.N_INIT,"k_grid":list(r1b.K_GRID),"alpha":r1b.ALPHA,"seed_root":SEED_ROOT,"geometries":g,"versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_cores_used":47}},"cells":{}}
    with Pool(47,initializer=_init,initargs=(cg,))as po:
        for pr in PR_TARGETS:
            k=f"pr{pr:g}_n{N}";cf=int(pr*1000)*7919+N*104729;t0=time.time();a=[(k,"gmm_R0.3",N,i,SEED_ROOT+cf+1000*i) for i in range(NDS)];rs=list(po.imap_unordered(r1b.one,a,chunksize=2))
            c=r1b.summarize(rs);c.update(pr_target=pr,n=N,family=f"gmm_R{R_TARGET}",wall_s=round(time.time()-t0,1));se["cells"][f"{k}_gmm_R{R_TARGET}"]=c;sr=c.get("separation_ratio",{})
            print(f"{k:12s} PR {c["pr_sphere"]["mean"]:6.2f}  R {sr.get("mean", float("nan")):.3f}  M {c["M"]["rej"]:.3f}  C {c["C"]["rej"]:.3f}  shift {c["shift"]["in_ambient_sd"]:+.3f} sd  {c["wall_s"]:.0f}s",flush=True)
            json.dump(se,open(o,"w"),indent=1)
    se["header"]["total_wall_s"]=round(time.time()-t,1);json.dump(se,open(o,"w"),indent=1);print(f"\ntotal wall {se["header"]["total_wall_s"] / 60:.1f} min");print(f"wrote {o}")
if __name__=="__main__":main()
