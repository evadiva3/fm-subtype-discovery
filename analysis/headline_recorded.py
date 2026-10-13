import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[v]='1'
import argparse,hashlib,json,platform,sys,time,traceback
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy,sklearn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'experiments/ds002748_mdd/propagated_null'))
import nullcal as nc
import ov_common as ov
G={}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def init(m,f,n):
    G.update(m=m,f=f,n=n)

def one(t):
    b,i,sd,ss,nd=t
    r=dict(block=b,index=i,data_seed=str(sd),reference_seed=str(ss))
    try:
        x=nc.l2_normalize(nc._draw(G['m'],G['f'],G['n'],np.random.default_rng(sd)))
        s,k,l=nc.best_silhouette(x,k_grid=ov.K_GRID,min_size=nc.min_cluster_guard(len(x)),n_init=5,random_state=i)
        r.update(data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),pr=nc.participation_ratio(x),observed=s,k=k,guard_failed=k is None,cluster_sizes=[] if l is None else np.bincount(l).tolist())
        if k is None:
            r.update(pM=1.,pC=1.)
        else:
            kw=dict(k_grid=ov.K_GRID,n_draws=nd,min_size=nc.min_cluster_guard(len(x)),n_init=5,seed=ss,observed=(s,k))
            for a,z in [('M',False),('C',True)]:
                q=nc.null_test(x,reproject=z,match_selection=z,**kw)
                r[a]=asdict(q)
                r['p'+a]=q.p
    except Exception:
        r['error']=traceback.format_exc()
    return r

def summary(rs):
    out=dict(n=len(rs),errors=sum('error' in r for r in rs),guard_failures=sum(r.get('guard_failed',False) for r in rs))
    if out['errors']:
        return out
    for a in ('M','C'):
        c=sum(r['p'+a]<.05 for r in rs)
        out[a]=dict(events=c,rate=c/len(rs),wilson95=list(nc._wilson(c,len(rs))),reference_guard_failures=sum(r.get(a,{}).get('n_degenerate',0) for r in rs))
    out.update(M_only=sum(r['pM']<.05<=r['pC'] for r in rs),C_only=sum(r['pC']<.05<=r['pM'] for r in rs),mean_pr=float(np.mean([r['pr'] for r in rs])))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--datasets',type=int,default=10000)
    p.add_argument('--draws',type=int,default=1000)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    if min(a.datasets,a.draws,a.workers)<1 or a.datasets%5:
        p.error('Positive counts required; datasets must be divisible by five.')
    a.output.mkdir(parents=True,exist_ok=True)
    e=R/'data/outputs/trained_fm_embeddings.npy'
    x=nc.l2_normalize(np.load(e).astype(np.float64))
    m,f=nc._gaussian_factor(x)
    root=2026092401 if a.pilot else 20260924
    cfg=dict(root_seed=root,experiment=71,n_datasets=a.datasets,n_draws=a.draws,n_blocks=5,n_init=5,k_grid=list(ov.K_GRID),alpha=.05,inequality='p < .05',embedding_shape=list(x.shape),input_sha256=sha(e),observed_guard_policy='retain as nonrejection with p=1',reference_guard_policy='legacy sentinel -1; retain all draws',fixed_k_guard='not applied in historical M implementation',seed_rule='ov.hashed_seed(root,71,0,block,index,purpose); purpose 0=data,1=reference',versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),source_sha256={str(q.relative_to(R)).replace('\\','/'):sha(q) for q in [Path(__file__),Path(nc.__file__),Path(ov.__file__)]})
    cp=a.output/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text(encoding='utf-8'))==cfg,'Configuration mismatch'
    else:
        cp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    jp=a.output/'records.jsonl'
    rs=[json.loads(l) for l in jp.read_text(encoding='utf-8').splitlines()] if jp.exists() else []
    done={(r['block'],r['index']) for r in rs}
    ts=[(b,i,ov.hashed_seed(root,71,0,b,i,0),ov.hashed_seed(root,71,0,b,i,1),a.draws) for b in range(5) for i in range(a.datasets//5) if (b,i) not in done]
    assert len({s for t in ts for s in t[2:4]})==2*len(ts)
    t0=time.time()
    with Pool(a.workers,initializer=init,initargs=(m,f,len(x))) as pool,jp.open('a',encoding='utf-8') as j:
        for r in pool.imap_unordered(one,ts,chunksize=1):
            j.write(json.dumps(r)+'\n')
            j.flush()
            rs.append(r)
            if len(rs)%50==0:
                print(f'{len(rs)}/{a.datasets}; {time.time()-t0:.0f}s',flush=True)
    out=dict(config=cfg,summary=summary(rs),blocks={str(b):summary([r for r in rs if r['block']==b]) for b in range(5)},records_sha256=sha(jp),execution_wall_s=time.time()-t0)
    (a.output/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['summary']),flush=True)
    if out['summary']['errors']:
        raise RuntimeError('Errors retained in records; analysis incomplete')

if __name__=='__main__':
    main()
