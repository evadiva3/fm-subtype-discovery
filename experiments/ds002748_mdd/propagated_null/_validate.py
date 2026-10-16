import os,time,json
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np,warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import geometry_surface as g

if __name__=="__main__":
    t0=time.time()
    with Pool(47)as pool:
        a=g.run_cell(28,119,3.0,base_seed=101,pool=pool)
        print(f"[A tau=3 ] PR={a['pr_mean']:.2f}  M={a['rej_M']:.3f}  C={a['rej_C']:.3f}  "
            f"M_only={a['M_only']} C_only={a['C_only']}  ({time.time()-t0:.0f}s)",flush=True)
        b=g.run_cell(28,119,3.0,base_seed=202,pool=pool)
        print(f"[B n=28  ] PR={b['pr_mean']:.2f}  M={b['rej_M']:.3f}  C={b['rej_C']:.3f}  "
            f"M_only={b['M_only']} C_only={b['C_only']}  ({time.time()-t0:.0f}s)",flush=True)
        c=g.run_cell(28,119,2.0,base_seed=101,pool=pool)
        print(f"[A tau=2 ] PR={c['pr_mean']:.2f}  M={c['rej_M']:.3f}  C={c['rej_C']:.3f}  "
            f"M_only={c['M_only']} C_only={c['C_only']}  ({time.time()-t0:.0f}s)",flush=True)
        d=g.run_cell(28,119,1.0,base_seed=101,pool=pool)
        print(f"[A tau=1 ] PR={d['pr_mean']:.2f}  M={d['rej_M']:.3f}  C={d['rej_C']:.3f}  "
            f"M_only={d['M_only']} C_only={d['C_only']}  ({time.time()-t0:.0f}s)",flush=True)
    json.dump({"A_tau3":a,"B_tau3":b,"A_tau2":c,"A_tau1":d},
        open("_validate.json","w"),indent=1)
    print(f"total {time.time()-t0:.0f}s")
