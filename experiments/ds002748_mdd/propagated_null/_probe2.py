import os, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, e1_grid as e

for n in (28,60,119,180,240):
    t=time.time(); tau,pr,ok,note=e.tune_tau(n,e.D,4.5); tt=time.time()-t
    t=time.time()
    r=[e.one((n,e.D,tau,i,1)) for i in range(3)]
    dt=(time.time()-t)/3
    bad=sum(1 for x in r if "pM" not in x)
    print(f"n={n:4d} tune {tt:5.1f}s tau={tau:9.4f} PR={pr:6.2f} ok={ok} {note} | "
          f"{dt:7.2f}s/dataset  bad={bad}", flush=True)
