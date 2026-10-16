#post hoc application of the S24 calibrated cutoffs to the observed fibromyalgia embedding
#uses the S24 implementation (499 references, five restarts) so observed p-values are on the calibrated scale
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import hashlib,json,platform,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"analysis"))
import mc_calibrated_sensitivity as s24
from summarize_detection import cutoff
nc,ov=s24.nc,s24.ov
CAL=R/"results/revision_20260925/mc_calibrated_sensitivity"
OUT=R/"reproducibility/audits/s24_observed_cohort.json"
NSEED=20

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def seed(j):
    return int.from_bytes(hashlib.sha256(f"20260925|86|{j}".encode()).digest()[:16],"big")

def observed():
    return nc.best_silhouette(s24._X,k_grid=ov.K_GRID,min_size=nc.min_cluster_guard(s24.N),n_init=5,random_state=0)

def one(j):
    s,k,_=observed()
    kw=dict(k_grid=ov.K_GRID,n_draws=499,min_size=nc.min_cluster_guard(s24.N),n_init=5,seed=seed(j),observed=(s,k))
    return {a:float(nc.null_test(s24._X,reproject=z,match_selection=z,**kw).p) for a,z in (("M",False),("C",True))}

def main():
    cal=[json.loads(l) for l in (CAL/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    cal=sorted([r for r in cal if r["group"]=="calibration"],key=lambda r:r["index"])
    assert len(cal)==1000
    cuts={a:cutoff(np.array([r["p"][a] for r in cal])) for a in ("M","C")}
    s,k,_=observed()
    with Pool(NSEED) as pool:
        ps=pool.map(one,range(NSEED))
    arms={}
    for a in ("M","C"):
        v=[p[a] for p in ps];t,rho=cuts[a]
        arms[a]=dict(cutoff=t,boundary_probability=rho,p=v,p_min=min(v),p_max=max(v),rejections_at_calibrated_cutoff=sum(x<t for x in v),ties_at_cutoff=sum(x==t for x in v))
    out=dict(analysis="post hoc: S24 cutoffs applied to the observed embedding",embedding=str(s24.EMB.relative_to(R)).replace("\\","/"),embedding_sha256=sha(s24.EMB),observed_silhouette=s,selected_k=k,reference_draws=499,reference_seeds=[str(seed(j)) for j in range(NSEED)],calibration_records_sha256=sha(CAL/"records.jsonl"),arms=arms,source_sha256={p:sha(R/p) for p in ("analysis/calibrated_cohort.py","analysis/mc_calibrated_sensitivity.py","analysis/summarize_detection.py","experiments/ds002748_mdd/propagated_null/nullcal.py","experiments/ds002748_mdd/propagated_null/ov_common.py")},versions=dict(python=platform.python_version(),numpy=np.__version__))
    OUT.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({a:{k:v for k,v in z.items() if k!="p"} for a,z in arms.items()},indent=2))

if __name__=="__main__":
    main()
