"""Outcome-motivated follow-up on independent observations; no primary overwrites."""
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
import angular_null as an
import angular_fourth as af
import nullcal_validation as nv
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(task):
    delta,q,index,cfg=task
    data_seed=nv.stream_seed(20260915,11,delta,0,index,0)
    rng=np.random.default_rng(data_seed);x=rng.standard_normal((60,20))
    x[:,0]+=delta*rng.choice([-1.,1.],size=60);x=nv.normalize(x)
    projection_seed=nv.stream_seed(20260915,8,100,q,0,2)
    projected=nv.normalize(x@an.fixed_projection(20,q,projection_seed))
    null_seed=nv.stream_seed(20260915,11,delta,q,index,3)
    base=dict(delta=delta,q=q,index=index,data_seed=str(data_seed),null_seed=str(null_seed),
        projection_seed=str(projection_seed),source_data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
    try:result=af.evaluate(projected,draws=cfg['draws'],seed=null_seed)
    except Exception:return dict(base,error=traceback.format_exc())
    return dict(base,**result)
def summarize(rows):
    assert not any('error' in r for r in rows),'Numerical failures: inspect all records'
    n=len(rows);count=sum(r['p']<=.05 for r in rows);ci=binomtest(count,n).proportion_ci(method='exact')
    return dict(n=n,count=count,rate=count/n,ci=[float(ci.low),float(ci.high)],numerical_failures=0)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/angular_v8_fourth')
    p.add_argument('--workers',type=int,default=36);p.add_argument('--datasets',type=int,default=200)
    p.add_argument('--draws',type=int,default=199);a=p.parse_args()
    cfg=dict(protocol='outcome-motivated independent follow-up',datasets=a.datasets,draws=a.draws,n=60,d=20,
        delta=[0,2,4],q=[5,10],python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(af.__file__),Path(an.__file__),Path(nv.__file__))})
    out=a.output;out.mkdir(exist_ok=True,parents=True);cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=cfg:raise RuntimeError('Configuration changed')
    cp.write_text(json.dumps(cfg,indent=2));cells={}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for delta in cfg['delta']:
            for q in cfg['q']:
                key=f'delta{delta}_q{q:02}';folder=out/key;folder.mkdir(exist_ok=True)
                jobs=[pool.submit(one,(delta,q,i,cfg)) for i in range(a.datasets) if not (folder/f'{i:05}.json').exists()]
                for job in as_completed(jobs):
                    row=job.result();(folder/f"{row['index']:05}.json").write_text(json.dumps(row,sort_keys=True))
                rows=[json.loads((folder/f'{i:05}.json').read_text()) for i in range(a.datasets)]
                cells[key]=dict(delta=delta,q=q,**summarize(rows));print(key,cells[key],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(config=cfg,cells=cells),indent=2))
if __name__=='__main__':main()
