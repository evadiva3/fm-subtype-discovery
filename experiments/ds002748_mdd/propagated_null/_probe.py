import os, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np, warnings
warnings.filterwarnings("ignore")
import geometry_surface as g

t0=time.time()
res=[g.one_dataset((28,119,3.0,i,101+1000*i)) for i in range(3)]
dt=time.time()-t0
print(f"3 datasets n=28 tau=3: {dt:.1f}s total, {dt/3:.2f}s per dataset (1 core)")
for r in res: print("   ", {k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
