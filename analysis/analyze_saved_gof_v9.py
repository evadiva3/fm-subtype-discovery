"""Apply fixed comparators to all 23 v8 saved representations; diagnostic only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,json,platform,sys,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy,sklearn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packaging'))
import angular_gof as gof
import angular_null as an
import nullcal_validation as nv
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(task):
    source,q,cfg,repo=task;raw=np.load(Path(repo)/source['path'],allow_pickle=False)
    ps=nv.stream_seed(20260915,9,0,q,0,2)
    projected=nv.normalize(raw@an.fixed_projection(raw.shape[1],q,ps))
    ns=nv.stream_seed(20260915,13,source['id'],q,0,3)
    bs=nv.stream_seed(20260915,13,source['id'],q,0,4)
    base=dict(source=source['path'],input_sha256=source['sha256'],id=source['id'],q=q,n=len(raw),d=raw.shape[1],
        projection_seed=str(ps),null_seed=str(ns),bootstrap_seed=str(bs),mean_resultant=float(np.linalg.norm(projected.mean(axis=0))))
    try:r=gof.evaluate(projected,draws=cfg['draws'],seed=ns,bootstrap_seed=bs,auxiliary=cfg['auxiliary'])
    except Exception:return dict(base,error=traceback.format_exc())
    return dict(base,**r)
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ROOT/'results/embeddings_v9')
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();repo=a.repo.resolve()
    sources=json.loads((repo/'results/embeddings_v8/config.json').read_text())['input_sources']
    assert len(sources)==23
    for s in sources:assert sha(repo/s['path'])==s['sha256']
    cfg=dict(version=gof.VERSION,draws=999,auxiliary=200,q=[5,10],input_sources=sources,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        protocol_sha256=sha(ROOT/'docs/PROTOCOL_V9_APPLIED.md'),
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(gof.__file__),Path(an.__file__),Path(nv.__file__))})
    out=a.output;out.mkdir(exist_ok=True,parents=True);cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=cfg:raise RuntimeError('Configuration changed')
    cp.write_text(json.dumps(cfg,indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(one,(s,q,cfg,str(repo))) for s in sources for q in cfg['q'] if not (out/f"source{s['id']:02}_q{q:02}.json").exists()]
        for job in as_completed(jobs):
            row=job.result();(out/f"source{row['id']:02}_q{row['q']:02}.json").write_text(json.dumps(row,indent=2))
            print(row['source'],row['q'],row.get('p',row.get('error')),flush=True)
    per_source=[]
    for s in sources:
        rows=[json.loads((out/f"source{s['id']:02}_q{q:02}.json").read_text()) for q in cfg['q']]
        assert all('p' in r for r in rows),'Numerical failure: inspect records'
        per_source.append(dict(s,q_results=rows,combined={arm:min(1.,2*min(r['p'][arm] for r in rows)) for arm in gof.ARMS}))
    (out/'summary.json').write_text(json.dumps(dict(config=cfg,sources=per_source,numerical_failures=0,
        interpretation='Descriptive model checks; shared training can violate independent-row assumptions.'),indent=2))
if __name__=='__main__':main()
