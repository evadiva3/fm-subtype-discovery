"""Prospective v8 calibration/power/restart study; see docs/PROTOCOL_V8.md."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.stats import binomtest
import sklearn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packaging'))
import angular_null as an
import nullcal_validation as nv

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def interval(k,n):
    ci=binomtest(k,n).proportion_ci(method='exact');return [float(ci.low),float(ci.high)]

def summarize(rows):
    failures=[r for r in rows if 'error' in r];valid=[r for r in rows if 'p' in r]
    if failures:raise RuntimeError(f'{len(failures)} numerical failures; inspect failure records before interpreting results')
    n=len(rows);result=dict(n=n,numerical_failures=0,
        raw_guard_failures=sum(r['raw_guard_failed'] for r in rows),
        whitened_guard_failures=sum(r['whitened_guard_failed'] for r in rows),arms={})
    assert len({r['source_data_seed'] for r in rows})==n
    for arm in ('C','T','W','N'):
        k=sum(r['p'][arm]<=.05 for r in rows)
        result['arms'][arm]=dict(count=k,rate=k/n,ci=interval(k,n))
    result['comparisons']={}
    for a,b in (('C','W'),('T','W'),('N','W')):
        left=sum(r['p'][a]<=.05<r['p'][b] for r in rows)
        right=sum(r['p'][b]<=.05<r['p'][a] for r in rows)
        result['comparisons'][a+'_vs_'+b]=dict(a_only=left,b_only=right,
            difference_b_minus_a=(right-left)/n,
            exact_p=float(binomtest(left,left+right,.5).pvalue) if left+right else 1.)
    result['max_fit_iterations']=max(r['shape_fit']['iterations'] for r in rows)
    result['max_fit_residual']=max(r['shape_fit']['whitened_moment_residual'] for r in rows)
    result['max_reference_fit_iterations']=max(r['max_reference_fit_iterations'] for r in rows)
    return result

def generate(spec,index):
    if spec['kind']=='calibration':
        rep,local=divmod(index,100)
        seed=nv.stream_seed(20260914,7,spec['id'],rep,local,0)
        x=np.random.default_rng(seed).normal(size=(spec['n'],spec['d']))@np.asarray(spec['population_factor'])
    else:
        seed=nv.stream_seed(20260915,8,spec['id'],0,index,0)
        rng=np.random.default_rng(seed);x=rng.normal(size=(spec['n'],spec['d']))
        x[:,0]+=spec['delta']*rng.choice([-1.,1.],size=spec['n'])
    return nv.normalize(x),seed

def one(task):
    spec,q,index,config=task
    x,data_seed=generate(spec,index)
    projection_seed=nv.stream_seed(config['root_seed'],8,spec.get('projection_id',spec['id']),q,0,2)
    proj=an.fixed_projection(spec['d'],q,projection_seed)
    projected=nv.normalize(x@proj)
    seed=nv.stream_seed(config['root_seed'],8,spec['id'],q,index,3)
    base=dict(cell=spec['name'],q=q,index=index,source_data_seed=str(data_seed),
              source_data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
              projection_seed=str(projection_seed),null_seed=str(seed))
    try:result=an.evaluate(projected,draws=config['draws'],seed=seed,n_init=config['n_init'])
    except Exception:return dict(base,error=traceback.format_exc())
    return dict(base,**result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--v7-root',type=Path,default=ROOT/'results/validation_v7')
    p.add_argument('--output',type=Path,default=ROOT/'results/angular_v8')
    p.add_argument('--datasets',type=int,default=200);p.add_argument('--draws',type=int,default=199)
    p.add_argument('--workers',type=int,default=36);p.add_argument('--n-init',type=int,default=5)
    p.add_argument('--q',nargs='+',type=int,default=[5,10]);p.add_argument('--cells',nargs='*')
    a=p.parse_args()
    if min(a.datasets,a.workers,a.n_init)<1 or a.draws<2:p.error('invalid count')
    pop=json.loads((a.v7_root/'populations.json').read_text())
    specs=[dict(c,kind='calibration') for c in pop]
    specs +=[dict(name=f'mixture_delta{delta}',id=100+j,projection_id=100,kind='power',n=60,d=20,delta=delta) for j,delta in enumerate([0,2,4])]
    if a.cells:
        if set(a.cells)-{s['name'] for s in specs}:p.error('unknown cell')
        specs=[s for s in specs if s['name'] in a.cells]
    if any(q>=min(s['n'],s['d']+1) or q<2 for s in specs for q in a.q):p.error('invalid projection dimension')
    cfg=dict(version=an.VERSION,root_seed=20260915,datasets=a.datasets,draws=a.draws,n_init=a.n_init,q=a.q,
             cells=[s['name'] for s in specs],threshold='p <= .05',
             python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
             v7_populations_sha256=sha(a.v7_root/'populations.json'),
             source_sha256={p.name:sha(p) for p in (Path(__file__),Path(an.__file__),Path(nv.__file__))})
    out=a.output;out.mkdir(parents=True,exist_ok=True);cpath=out/'config.json'
    if cpath.exists() and json.loads(cpath.read_text())!=cfg:raise RuntimeError('checkpoint configuration changed')
    cpath.write_text(json.dumps(cfg,indent=2));(out/'populations.json').write_text(json.dumps(specs,indent=1))
    summaries={};started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for spec in specs:
            for q in a.q:
                name=spec['name']+f'_q{q:02}';folder=out/name;folder.mkdir(exist_ok=True)
                pending=[i for i in range(a.datasets) if not (folder/f'{i:05}.json').exists()]
                jobs={pool.submit(one,(spec,q,i,cfg)):i for i in pending}
                done=a.datasets-len(pending)
                for future in as_completed(jobs):
                    row=future.result();(folder/f"{row['index']:05}.json").write_text(json.dumps(row,sort_keys=True))
                    if 'error' in row:raise RuntimeError(row['error'])
                    done+=1
                    if done%25==0:print(f'{name}: {done}/{a.datasets}, elapsed {time.monotonic()-started:.0f}s',flush=True)
                rows=[json.loads((folder/f'{i:05}.json').read_text()) for i in range(a.datasets)]
                summary=summarize(rows);summary.update(kind=spec['kind'],q=q,source_cell=spec['name'],n_subjects=spec['n'],d=spec['d'])
                (folder/'summary.json').write_text(json.dumps(summary,indent=2));summaries[name]=summary
                print(name,{arm:round(v['rate'],4) for arm,v in summary['arms'].items()},flush=True)
    (out/'summary.json').write_text(json.dumps(dict(config=cfg,cells=summaries),indent=2))

if __name__=='__main__':main()
