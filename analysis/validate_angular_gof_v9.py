"""Fixed v9 comparative study; all cells and methods are retained."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,json,platform,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import scipy,sklearn
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packaging'))
import angular_gof as gof
import angular_null as an
import nullcal_validation as nv
FAMILIES=('gaussian','anisotropic','two_modes','six_modes','offset','shape_mixture')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True))
def generate(family,n,index):
    seed=nv.stream_seed(20260915,12,FAMILIES.index(family),n,index,0)
    rng=np.random.default_rng(seed);x=rng.standard_normal((n,20))
    if family=='anisotropic':x*=np.exp(-np.arange(20)/6)
    elif family=='two_modes':x[:,0]+=4*rng.choice([-1.,1.],size=n)
    elif family=='six_modes':
        axis=rng.integers(0,3,size=n);x[np.arange(n),axis]+=4*rng.choice([-1.,1.],size=n)
    elif family=='offset':x[:,0]+=1
    elif family=='shape_mixture':
        axis=rng.integers(0,2,size=n);x[np.arange(n),axis]*=4
    return nv.normalize(x),seed
def projection(q):
    if q==20:return np.eye(20),None
    seed=nv.stream_seed(20260915,8,100,q,0,2)
    return an.fixed_projection(20,q,seed),str(seed)
def prepare(family,n,q,index):
    x,seed=generate(family,n,index);proj,ps=projection(q)
    null=nv.stream_seed(20260915,12,FAMILIES.index(family),n,index,100+q)
    boot=nv.stream_seed(20260915,12,FAMILIES.index(family),n,index,200+q)
    base=dict(family=family,n=n,d=20,q=q,index=index,source_data_seed=str(seed),
        projection_seed=ps,null_seed=str(null),bootstrap_seed=str(boot),
        source_data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
    return nv.normalize(x@proj),base
def one(task):
    family,n,q,index,cfg=task;x,base=prepare(family,n,q,index)
    try:result=gof.evaluate(x,draws=cfg['draws'],seed=int(base['null_seed']),bootstrap_seed=int(base['bootstrap_seed']),auxiliary=cfg['auxiliary'])
    except Exception:return dict(base,error=traceback.format_exc())
    return dict(base,**result)
def interval(k,n):
    ci=binomtest(k,n).proportion_ci(method='exact');return [float(ci.low),float(ci.high)]
def summarize(rows):
    failures=sum('error' in r for r in rows)
    if failures:raise RuntimeError(f'{failures} failed analyses; inspect records before interpreting')
    n=len(rows);arms={}
    for a in gof.ARMS:
        count=sum(r['p'][a]<=.05 for r in rows)
        arms[a]=dict(count=count,rate=count/n,ci=interval(count,n))
    pairs={}
    for a,b in [('W','D4'),('W','K'),('W','E'),('K','E')]:
        left=sum(r['p'][a]<=.05<r['p'][b] for r in rows);right=sum(r['p'][b]<=.05<r['p'][a] for r in rows)
        pairs[a+'_vs_'+b]=dict(a_only=left,b_only=right,difference_b_minus_a=(right-left)/n,
            exact_p=float(binomtest(left,left+right,.5).pvalue) if left+right else 1.)
    return dict(n_datasets=n,arms=arms,pairs=pairs,numerical_failures=0,
        guard_failures=sum(r['guard_failed'] for r in rows),
        max_observed_residual=max(r['shape_fit']['whitened_moment_residual'] for r in rows),
        max_reference_residual=max(r['max_reference_residual'] for r in rows))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/angular_v9')
    p.add_argument('--workers',type=int,default=36);p.add_argument('--datasets',type=int,default=200)
    p.add_argument('--draws',type=int,default=199);p.add_argument('--families',nargs='+',default=list(FAMILIES))
    p.add_argument('--n',nargs='+',type=int,default=[60,240]);p.add_argument('--q',nargs='+',type=int,default=[5,10,20])
    a=p.parse_args()
    if set(a.families)-set(FAMILIES) or min(a.n)<=max(a.q) or set(a.q)-{5,10,20} or min(a.datasets,a.workers)<1 or a.draws<2:p.error('invalid setting')
    cfg=dict(version=gof.VERSION,datasets=a.datasets,draws=a.draws,auxiliary=200,
        n=a.n,q=a.q,families=a.families,arms=list(gof.ARMS),threshold='p <= .05',
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        protocol_sha256=sha(ROOT/'docs/PROTOCOL_V9.md'),
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(gof.__file__),Path(an.__file__),Path(nv.__file__))})
    out=a.output;out.mkdir(exist_ok=True,parents=True);cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=cfg:raise RuntimeError('Configuration changed')
    dump(cp,cfg);cells={};started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for family in a.families:
            for n in a.n:
                for q in a.q:
                    key=f'{family}_n{n}_q{q:02}';folder=out/key;folder.mkdir(exist_ok=True)
                    pending=[i for i in range(a.datasets) if not (folder/f'{i:05}.json').exists()]
                    jobs=[pool.submit(one,(family,n,q,i,cfg)) for i in pending];done=a.datasets-len(pending)
                    for job in as_completed(jobs):
                        row=job.result();dump(folder/f"{row['index']:05}.json",row);done+=1
                        if done%50==0:print(f'{key}: {done}/{a.datasets}, elapsed {time.monotonic()-started:.0f}s',flush=True)
                    rows=[json.loads((folder/f'{i:05}.json').read_text()) for i in range(a.datasets)]
                    cells[key]=dict(family=family,n=n,q=q,kind='null' if family in FAMILIES[:2] else 'alternative',**summarize(rows))
                    dump(folder/'summary.json',cells[key]);print(key,{k:v['rate'] for k,v in cells[key]['arms'].items()},flush=True)
    projections={str(q):dict(first_three_axis_retention=np.sum(projection(q)[0][:3]**2,axis=1).tolist(),seed=projection(q)[1]) for q in a.q}
    dump(out/'summary.json',dict(config=cfg,cells=cells,projections=projections,elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':main()
