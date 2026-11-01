"""Independent follow-up of a fixed even/odd kernel decomposition."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,json,platform,sys,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy,sklearn
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packaging'))
import angular_balanced as ab
import angular_null as an
import nullcal_validation as nv
import validate_angular_gof_v9 as primary
FAMILIES=primary.FAMILIES
def generate(family,n,index):
    seed=nv.stream_seed(20260915,14,FAMILIES.index(family),n,index,0)
    rng=np.random.default_rng(seed);x=rng.standard_normal((n,20))
    if family=='anisotropic':x*=np.exp(-np.arange(20)/6)
    elif family=='two_modes':x[:,0]+=4*rng.choice([-1.,1.],size=n)
    elif family=='six_modes':
        axis=rng.integers(0,3,size=n);x[np.arange(n),axis]+=4*rng.choice([-1.,1.],size=n)
    elif family=='offset':x[:,0]+=1
    elif family=='shape_mixture':
        axis=rng.integers(0,2,size=n);x[np.arange(n),axis]*=4
    return nv.normalize(x),seed
def prepare(family,n,q,index):
    x,seed=generate(family,n,index);proj,ps=primary.projection(q)
    ns=nv.stream_seed(20260915,14,FAMILIES.index(family),n,index,100+q)
    return nv.normalize(x@proj),dict(family=family,n=n,q=q,index=index,source_data_seed=str(seed),null_seed=str(ns),
        projection_seed=ps,source_data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
def one(task):
    family,n,q,index,cfg=task;x,base=prepare(family,n,q,index)
    try:r=ab.evaluate(x,draws=cfg['draws'],seed=int(base['null_seed']))
    except Exception:return dict(base,error=traceback.format_exc())
    return dict(base,**r)
def summarize(rows):
    if any('error' in r for r in rows):raise RuntimeError('Numerical failures: inspect all records')
    n=len(rows);arms={}
    for arm in ab.ARMS:
        k=sum(r['p'][arm]<=.05 for r in rows);arms[arm]=dict(count=k,rate=k/n,ci=primary.interval(k,n))
    left=sum(r['p']['K']<=.05<r['p']['KB'] for r in rows);right=sum(r['p']['KB']<=.05<r['p']['K'] for r in rows)
    return dict(n_datasets=n,arms=arms,numerical_failures=0,K_vs_KB=dict(a_only=left,b_only=right,
        difference_b_minus_a=(right-left)/n,exact_p=float(binomtest(left,left+right,.5).pvalue) if left+right else 1.))
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    cfg=dict(datasets=200,draws=199,n=[60,240],q=[5,10,20],families=list(FAMILIES),arms=list(ab.ARMS),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        protocol_sha256=primary.sha(ROOT/'docs/PROTOCOL_V9_KERNEL_FOLLOWUP.md'),
        source_sha256={p.name:primary.sha(p) for p in (Path(__file__),Path(ab.__file__),Path(an.__file__),Path(nv.__file__),Path(primary.__file__),Path(primary.gof.__file__))})
    out=ROOT/'results/angular_v9_balanced';out.mkdir(exist_ok=True,parents=True);cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=cfg:raise RuntimeError('Configuration changed')
    primary.dump(cp,cfg);cells={}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for family in cfg['families']:
            for n in cfg['n']:
                for q in cfg['q']:
                    key=f'{family}_n{n}_q{q:02}';folder=out/key;folder.mkdir(exist_ok=True)
                    jobs=[pool.submit(one,(family,n,q,i,cfg)) for i in range(cfg['datasets']) if not (folder/f'{i:05}.json').exists()]
                    for job in as_completed(jobs):
                        r=job.result();primary.dump(folder/f"{r['index']:05}.json",r)
                    rows=[json.loads((folder/f'{i:05}.json').read_text()) for i in range(cfg['datasets'])]
                    cells[key]=dict(family=family,n=n,q=q,kind='null' if family in FAMILIES[:2] else 'alternative',**summarize(rows))
                    primary.dump(folder/'summary.json',cells[key]);print(key,{a:v['rate'] for a,v in cells[key]['arms'].items()},flush=True)
    primary.dump(out/'summary.json',dict(config=cfg,cells=cells))
if __name__=='__main__':main()
