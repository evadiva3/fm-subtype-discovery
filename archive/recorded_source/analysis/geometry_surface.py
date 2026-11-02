"""Fresh, completely recorded follow-up to the historical S9 geometry sweeps."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,platform,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import scipy,sklearn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/ds002748_mdd/propagated_null'))
import nullcal as nc
import ov_common as ov
CELLS=[('A',28,tau) for tau in (.5,1.,2.,3.,5.,8.,15.,40.,None)]+[('B',n,3.) for n in (28,50,100,200)]

def one(task):
    cell,index,draws=task
    sweep,n,tau=CELLS[cell]
    data_seed=ov.hashed_seed(20260923,62,cell,0,index,0)
    draw_seed=ov.hashed_seed(20260923,62,cell,0,index,1)
    row=dict(cell=cell,sweep=sweep,n=n,d=119,tau=tau,index=index,data_seed=str(data_seed),draw_seed=str(draw_seed))
    try:
        rng=np.random.default_rng(data_seed)
        lam=np.ones(119) if tau is None else np.exp(-np.arange(119)/tau)
        rotation=np.linalg.qr(rng.standard_normal((119,119)))[0]
        x=nc.l2_normalize(rng.standard_normal((n,119))@(np.sqrt(lam)[:,None]*rotation.T))
        row['source_sha256']=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        row['pr']=float(nc.participation_ratio(x))
        score,k,_=nc.best_silhouette(x,n_init=5,random_state=index)
        row.update(guard_failed=k is None,k=k,observed=float(score))
        if k is None:
            row['pM']=row['pC']=1.
            return row
        kw=dict(k_grid=(2,3,4,5,6),n_draws=draws,n_init=5,seed=draw_seed,observed=(score,k))
        row['pM']=float(nc.null_test(x,reproject=False,match_selection=False,**kw).p)
        row['pC']=float(nc.null_test(x,reproject=True,match_selection=True,**kw).p)
    except Exception:
        row['error']=traceback.format_exc()
    return row

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/revision_20260923/geometry_surface_fresh.json')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--datasets',type=int,default=120)
    parser.add_argument('--draws',type=int,default=200)
    a=parser.parse_args()
    if a.output.exists() or a.output.with_suffix('.records.jsonl').exists():parser.error('Choose a fresh output path.')
    if min(a.workers,a.datasets,a.draws)<1:parser.error('All counts must be positive.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tasks=[(c,i,a.draws) for c in range(len(CELLS)) for i in range(a.datasets)]
    started=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,a.output.with_suffix('.records.jsonl').open('w') as journal:
        jobs=[pool.submit(one,t) for t in tasks]
        for job in as_completed(jobs):
            r=job.result();rows.append(r);journal.write(json.dumps(r)+'\n');journal.flush()
            if len(rows)%100==0:print(f'{len(rows)}/{len(tasks)} complete, {time.time()-started:.0f}s',flush=True)
    out=dict(config=dict(root_seed=20260923,experiment=62,datasets=a.datasets,draws=a.draws,n_init=5,alpha=.05,
        inequality='p < .05',observed_guard_policy='count as nonrejection with p=1',
        seed_rule='ov.hashed_seed(root,experiment,cell,0,index,purpose), purpose 0=data 1=draws',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(nc.__file__),Path(ov.__file__))}),
        records=sorted(rows,key=lambda r:(r['cell'],r['index'])),cells={})
    errors=[r for r in rows if 'error' in r]
    if errors:
        a.output.with_suffix('.failed.json').write_text(json.dumps(out,indent=2));raise RuntimeError(errors[0]['error'])
    for c,(sweep,n,tau) in enumerate(CELLS):
        rr=[r for r in rows if r['cell']==c];size=len(rr)
        cell=dict(sweep=sweep,n=n,tau=tau,n_datasets=size,pr_mean=float(np.mean([r['pr'] for r in rr])),guard_failures=sum(r['guard_failed'] for r in rr))
        for arm in ('M','C'):
            k=sum(r['p'+arm]<.05 for r in rr)
            cell[arm]=dict(events=k,rate=k/size,wilson95=list(nc._wilson(k,size)))
        cell['M_only']=sum(r['pM']<.05<=r['pC'] for r in rr)
        cell['C_only']=sum(r['pC']<.05<=r['pM'] for r in rr)
        out['cells'][str(c)]=cell
    out['total_wall_s']=time.time()-started
    a.output.write_text(json.dumps(out,indent=2))
    print('Completed',len(rows),'analyses;',round(out['total_wall_s'],1),'seconds',flush=True)

if __name__=='__main__':main()
