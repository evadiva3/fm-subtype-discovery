#Phase 1D verification: the C-only rejections found in the regenerated Sweep A.
import json
import os
import platform
import sys
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nullcal as nc
import ov_common as ov
D=119;K_GRID=(2,3,4,5,6);ALPHA=0.05;N_INIT=5;NDS=120;SEED_A=3100000;SEED_B=3200000;DEEP=2000
CELLS=[("A",28,t,SEED_A+int(0 if np.isinf(t) else t*977)) for t in[0.5,1.0,2.0,3.0,5.0,8.0,15.0,40.0,float("inf")]]+[("B",n,3.0,SEED_B+n*977) for n in[28,50,100,200]]
def run(args):
    n,t,b,i,nd=args;l=ov.spec_exp(D,t);X=ov.gen(n,D,l,b+1000*i);g=nc.min_cluster_guard(n);s,k,_=nc.best_silhouette(X,k_grid=K_GRID,min_size=g,n_init=N_INIT,random_state=i)
    if k is None:return None
    kw=dict(k_grid=K_GRID,n_draws=nd,min_size=g,n_init=N_INIT,seed=b+1000*i+7,observed=(s,k))
    return(i,float(nc.null_test(X,reproject=False,match_selection=False,**kw).p),float(nc.null_test(X,reproject=True,match_selection=True,**kw).p))
def main():
    s=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");o1=ov.ROOT/"results"/"phase1"/s;o1.mkdir(parents=True,exist_ok=True);o=o1/"r1d_conly_check.json"
    import scipy
    import sklearn
    se={"header":{"experiment":"R1D C-only verification","purpose":"re-test every C-only discordance of the regenerated sweeps at 2,000 draws","written_utc":s,"d":D,"n_datasets":NDS,"n_draws_screen":200,"n_draws_deep":DEEP,"alpha":ALPHA,"n_init":N_INIT,"k_grid":list(K_GRID),"seed_base_sweep_A":SEED_A,"seed_base_sweep_B":SEED_B,"seed_rule":"dataset i uses seed_base + 1000*i; draws use seed_base + 1000*i + 7","protocol":"Section 3.3 and S3.1 of the article","versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform()}},"cells":{}}
    tc=tv=0
    with Pool(47)as po:
        for sp,n,t,b in CELLS:
            k=f"{sp}_n{n}_tau{("inf" if np.isinf(t) else f'{t:g}')}";rs=[r for r in po.imap_unordered(run,[(n,t,b,i,200) for i in range(NDS)],chunksize=1) if r]
            dc=sorted([r for r in rs if r[2]<ALPHA<=r[1]]);rc={"sweep":sp,"n":n,"tau":None if np.isinf(t) else t,"n_datasets":len(rs),"C_only_at_200":len(dc),"discordant":[]}
            if dc:
                d={r[0]:r for r in po.imap_unordered(run,[(n,t,b,i,DEEP) for i,_,_ in dc],chunksize=1) if r}
                for i,pM,pC in dc:
                    dM,dC=(d[i][1],d[i][2]);p=bool(dC<ALPHA<=dM)
                    rc["discordant"].append({"dataset":i,"pM_200":pM,"pC_200":pC,"pM_2000":dM,"pC_2000":dC,"steps_from_alpha_200":round(abs(pC-ALPHA)*201,2),"persists_at_2000":p});tv+=p
            tc+=len(dc);se["cells"][k]=rc;print(f"{k:22s} C-only {len(dc)}",flush=True);json.dump(se,open(o,"w"),indent=1)
    se["header"]["total_C_only"]=tc;se["header"]["total_persisting_at_2000"]=tv;json.dump(se,open(o,"w"),indent=1);print(f'\ntotal C-only at 200 draws: {tc}')
    print(f"surviving at {DEEP} draws:   {tv}");print(f"wrote {o}")
if __name__=="__main__":main()
