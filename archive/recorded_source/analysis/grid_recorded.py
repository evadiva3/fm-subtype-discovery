"""S26: fully recorded rerun of the 30-cell conditioning x sample-size grid; see docs/FOLLOWUP_PROTOCOL_S24_S26_20260925.md."""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[v]='1'
import argparse,hashlib,json,platform,sys,time,traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy,sklearn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'experiments/ds002748_mdd/propagated_null'))
import nullcal as nc
import ov_common as ov
EXP=80
PROTOCOL=R/'docs/FOLLOWUP_PROTOCOL_S24_S26_20260925.md'
HIST=R/'results/overnight/e1_grid.json'
_h=json.loads(HIST.read_text(encoding='utf-8'))['cells']
CELLS=[dict(key=k,n=c['n'],pr_target=c['pr_target'],tau=c['tau']) for k,c in sorted(_h.items(),key=lambda kc:(kc[1]['pr_target'],kc[1]['n']))]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def one(t):
    root,cell,i,nd=t
    c=CELLS[cell];n=c['n']
    ds=ov.hashed_seed(root,EXP,cell,0,i,0);rs=ov.hashed_seed(root,EXP,cell,0,i,1)
    r=dict(cell=cell,key=c['key'],n=n,pr_target=c['pr_target'],tau=c['tau'],index=i,data_seed=str(ds),reference_seed=str(rs))
    try:
        rng=np.random.default_rng(ds)
        lam=np.exp(-np.arange(119)/c['tau'])
        rot=np.linalg.qr(rng.standard_normal((119,119)))[0]
        x=nc.l2_normalize(rng.standard_normal((n,119))@(np.sqrt(lam)[:,None]*rot.T))
        s,k,_=nc.best_silhouette(x,k_grid=ov.K_GRID,min_size=nc.min_cluster_guard(n),n_init=5,random_state=i)
        r.update(source_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),pr=float(nc.participation_ratio(x)),observed=float(s),k=k,guard_failed=k is None)
        if k is None:
            r.update(pM=1.,pC=1.)
        else:
            kw=dict(k_grid=ov.K_GRID,n_draws=nd,min_size=nc.min_cluster_guard(n),n_init=5,seed=rs,observed=(s,k))
            r['pM']=float(nc.null_test(x,reproject=False,match_selection=False,**kw).p)
            r['pC']=float(nc.null_test(x,reproject=True,match_selection=True,**kw).p)
    except Exception:
        r['error']=traceback.format_exc()
    return r

def summary(rows):
    out={}
    for c,cell in enumerate(CELLS):
        rr=[r for r in rows if r['cell']==c];m=len(rr)
        o=dict(key=cell['key'],n=cell['n'],pr_target=cell['pr_target'],tau=cell['tau'],n_datasets=m,pr_mean=float(np.mean([r['pr'] for r in rr])),guard_failures=sum(r['guard_failed'] for r in rr))
        for a in ('M','C'):
            e=sum(r['p'+a]<.05 for r in rr);o[a]=dict(events=e,rate=e/m,wilson95=list(nc._wilson(e,m)))
        o['M_only']=sum(r['pM']<.05<=r['pC'] for r in rr);o['C_only']=sum(r['pC']<.05<=r['pM'] for r in rr)
        out[str(c)]=o
    return out

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--workers',type=int,default=16)
    a.add_argument('--pilot',action='store_true')
    a=a.parse_args()
    root=2026092502 if a.pilot else 20260925
    nd=2 if a.pilot else 400
    cfg=dict(root_seed=root,experiment=EXP,cells=CELLS,datasets_per_cell=nd,draws=200,n_init=5,k_grid=list(ov.K_GRID),inequality='p < .05',observed_guard_policy='count as nonrejection with p=1',tau_source=str(HIST.relative_to(R)).replace('\\','/'),tau_source_sha256=sha(HIST),protocol_sha256=sha(PROTOCOL),source_sha256={str(f.relative_to(R)).replace('\\','/'):sha(f) for f in (Path(__file__),Path(nc.__file__),Path(ov.__file__))},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    a.output.mkdir(parents=True,exist_ok=True)
    cp=a.output/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text(encoding='utf-8'))==cfg,'Configuration mismatch'
    else:
        cp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    jp=a.output/'records.jsonl'
    rows=[json.loads(l) for l in jp.read_text(encoding='utf-8').splitlines()] if jp.exists() else []
    done={(r['cell'],r['index']) for r in rows}
    # Largest cells first so the pool stays busy.
    ts=sorted([(root,c,i,cfg['draws']) for c in range(len(CELLS)) for i in range(nd) if (c,i) not in done],key=lambda t:-CELLS[t[1]]['n'])
    t0=time.time()
    with Pool(a.workers) as pool,jp.open('a',encoding='utf-8') as j:
        for r in pool.imap_unordered(one,ts,chunksize=1):
            j.write(json.dumps(r)+'\n');j.flush();rows.append(r)
            if len(rows)%500==0:
                print(f'{len(rows)}/{len(CELLS)*nd}; {time.time()-t0:.0f}s',flush=True)
    errors=sum('error' in r for r in rows)
    out=dict(config=cfg,records_sha256=sha(jp),errors=errors,execution_wall_s=time.time()-t0)
    if not errors:
        out['cells']=summary(rows)
    (a.output/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(dict(n=len(rows),errors=errors,wall=out['execution_wall_s'])),flush=True)
    if errors:
        raise RuntimeError('Errors retained; do not summarize as completed')

if __name__=='__main__':
    main()
