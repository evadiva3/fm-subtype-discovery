#same sweep with the mixture on a minor axis
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_v,"1")
import numpy as np
from scipy.stats import binomtest
_R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_R/"packaging"));sys.path.insert(0,str(_R/"experiments"/"ds002748_mdd"/"propagated_null"))
import angular_gof as gof
import angular_null as an
import nullcal_validation as nv
import ov_common as ov
OUT=_R/"results"/"shape_aware_minor_axis.json";SHAPE=_R/"results"/"overnight"/"e2_spectrum_shape.json";N=28;D=ov.D;NDS=200;DRAWS=199;AUXILIARY=200;QS=(5,10);DELTA=4.0;ROOT=20260918;EXPERIMENT=42
N_WORKERS=47;DELTAS=(0.0,4.0,8.0);AXIS=5;CELLS=[("delta%g"%x,"two_block_pr4.5",x) for x in DELTAS];_LAM={}
def spectrum(key):
    c=json.loads(SHAPE.read_text())["cells"][key]
    if c["family"]=="exp":return (ov.spec_exp(D,float(c["param_value"])),c)
    m,e=str(c["param_value"]).split("/");return (ov.spec_two_block(D,int(m),float(e)),c)
def _init(lams):_LAM.update(lams)
def one(task):
    n,q,i=task
    try:
        l=_LAM[n];s=nv.stream_seed(ROOT,EXPERIMENT,CELL_ID[n],q,i,0);rg=np.random.default_rng(s);x=rg.standard_normal((N,D))@(np.sqrt(l)[:,None]*np.linalg.qr(rg.standard_normal((D,D)))[0].T)
        p=CELL_PLANTED[n]
        if p:x[:,AXIS]+=p*rg.choice([-1.0,1.0],size=N)*np.sqrt(l[AXIS])
        xn=nv.normalize(x);pd=nv.stream_seed(ROOT,EXPERIMENT,99,q,0,2);pj=an.fixed_projection(D,q,pd);xp=nv.normalize(xn@pj);nd=nv.stream_seed(ROOT,EXPERIMENT,CELL_ID[n],q,i,100)
        b=nv.stream_seed(ROOT,EXPERIMENT,CELL_ID[n],q,i,200);r=gof.evaluate(xp,draws=DRAWS,seed=int(nd),bootstrap_seed=int(b),auxiliary=AUXILIARY)
        return dict(cell=n,q=q,index=i,planted=p,source_data_seed=str(s),null_seed=str(nd),projection_seed=str(pd),source_data_sha256=hashlib.sha256(xn.astype("<f8").tobytes()).hexdigest(),p=r["p"])
    except Exception:return dict(cell=n,q=q,index=i,error=traceback.format_exc())
CELL_ID={name:k for k,(name,_,_) in enumerate(CELLS)};CELL_PLANTED={name:delta for name,_,delta in CELLS}
def interval(k,n):
    ci=binomtest(k,n).proportion_ci(method="exact");return [float(ci.low),float(ci.high)]
def main():
    import scipy
    import sklearn
    ls={};m={}
    for n,k,p in CELLS:
        l,cl=spectrum(k);ls[n]=l;m[n]=dict(spectrum_key=k,family=cl["family"],param_value=cl["param_value"],planted_delta=p,planted_axis=AXIS,archived_pr_mean=cl["pr_mean"])
    t=[(n,q,i) for n,_,_ in CELLS for q in QS for i in range(NDS)];print("%d analyses: %d cells x %d projections x %d datasets, n = %d, d = %d"%(len(t),len(CELLS),len(QS),NDS,N,D),flush=True)
    t0=time.time()
    with Pool(N_WORKERS,initializer=_init,initargs=(ls,)) as pl:rs=list(pl.imap_unordered(one,t,chunksize=1))
    e=[r for r in rs if "error" in r]
    if e:raise RuntimeError(e[0]["error"])
    o=dict(experiment="shape-aware reference at the article's own geometry",written_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),n=N,d=D,n_datasets=NDS,draws=DRAWS,auxiliary=AUXILIARY,q=list(QS),planted_delta=DELTA,alpha=0.05,arms=list(gof.ARMS),cells=m,seed_rule="nullcal_validation.stream_seed(%d, %d, cell, q, dataset, purpose)"%(ROOT,EXPERIMENT),versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"platform":platform.platform(),"n_workers":N_WORKERS},results={})
    h=set()
    for n,_,_ in CELLS:
        for q in QS:
            s=[r for r in rs if r["cell"]==n and r["q"]==q];h|={r["source_data_sha256"] for r in s};a1={}
            for a in gof.ARMS:
                c=sum((r["p"][a]<=0.05 for r in s));a1[a]=dict(count=c,rate=c/len(s),ci=interval(c,len(s)))
            o["results"]["%s_q%02d"%(n,q)]=dict(cell=n,q=q,n_datasets=len(s),kind="alternative" if CELL_PLANTED[n] else "null",delta=CELL_PLANTED[n],arms=a1)
            print("[%s q=%d] "%(n,q)+"  ".join(("%s %.1f%% [%.1f, %.1f]"%(a,100*a1[a]["rate"],100*a1[a]["ci"][0],100*a1[a]["ci"][1]) for a in gof.ARMS)),flush=True)
    o["distinct_source_datasets"]=len(h);o["total_wall_s"]=round(time.time()-t0,1);OUT.write_text(json.dumps(o,indent=1))
    print("distinct source datasets %d, wall %.0f min"%(len(h),o["total_wall_s"]/60));print("wrote %s"%OUT)
if __name__=="__main__":main()
