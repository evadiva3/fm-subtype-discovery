"""S24: sensitivity of M and C at calibrated size; see docs/FOLLOWUP_PROTOCOL_S24_S26_20260925.md."""
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
EXP=78
G=('calibration','null')+tuple(f'{f}{d}' for f in ('two','three') for d in ('0.5','1.0','1.5','2.0'))
PROTOCOL=R/'docs/FOLLOWUP_PROTOCOL_S24_S26_20260925.md'
EMB=R/'data/outputs/trained_fm_embeddings.npy'
_X=nc.l2_normalize(np.load(EMB).astype(np.float64))
MEAN,FACTOR=nc._gaussian_factor(_X)
N=len(_X)
_lam,_vec=np.linalg.eigh(FACTOR.T@FACTOR)
BASIS=(np.sqrt(np.clip(_lam[::-1][:2],0,None))*_vec[:,::-1][:,:2]).T  # rows m1, m2 (one SD along each leading axis)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sd(root,*a):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(root,EXP,*a))).encode()).digest()[:16],'big')

def means(name,perm):
    if name in ('calibration','null'):
        return None,None
    fam=name.rstrip('0123456789.');de=float(name[len(fam):])
    if fam=='two':
        c=np.array([[1.,0.],[-1.,0.]]);sizes=[N-N//2,N//2]
    else:
        c=np.array([[np.cos(t),np.sin(t)] for t in (0.,2*np.pi/3,4*np.pi/3)]);sizes=[N-2*(N//3),N//3,N//3]
    M=de*c@BASIS
    lab=np.repeat(np.arange(len(sizes)),sizes)[np.argsort(np.argsort(perm))]
    w=np.bincount(lab,minlength=len(M))/N
    return M-w@M,lab

def one(t):
    root,g,i,nd=t
    sid=0 if g==0 else 1
    ds=sd(root,0,sid,i)
    r=dict(spectrum=0,q=0,group=G[g],index=i,data_seed=str(ds))
    try:
        rng=np.random.default_rng(ds)
        x=nc._draw(MEAN,FACTOR,N,rng)
        perm=rng.permutation(N)
        M,lab=means(G[g],perm)
        if M is not None:
            x=x+M[lab]
        x=nc.l2_normalize(x)
        rs=sd(root,2,g,i)
        u=float(np.random.default_rng(sd(root,4,g,i)).uniform())
        s,k,l=nc.best_silhouette(x,k_grid=ov.K_GRID,min_size=nc.min_cluster_guard(N),n_init=5,random_state=i)
        r.update(labels=None if lab is None else lab.tolist(),source_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),observed=s,selected_k=k,guard_failed=k is None,reference_seed=str(rs),boundary_uniform=u,standardized_between=0. if M is None else float(np.sum((np.bincount(lab)/N)[:,None]*(M@np.linalg.pinv(BASIS))**2)))
        if k is None:
            r.update(p={'M':1.,'C':1.},reference_guard_failures=0)
        else:
            kw=dict(k_grid=ov.K_GRID,n_draws=nd,min_size=nc.min_cluster_guard(N),n_init=5,seed=rs,observed=(s,k))
            res={a:nc.null_test(x,reproject=z,match_selection=z,**kw) for a,z in (('M',False),('C',True))}
            r.update(p={a:float(v.p) for a,v in res.items()},reference_guard_failures=int(res['C'].n_degenerate))
    except Exception:
        r['error']=traceback.format_exc()
    return r

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--workers',type=int,default=16)
    a.add_argument('--pilot',action='store_true')
    a=a.parse_args()
    root=2026092501 if a.pilot else 20260925
    ns=[4]*len(G) if a.pilot else [1000,1000]+[250]*(len(G)-2)
    cfg=dict(root_seed=root,experiment=EXP,n=N,d=int(_X.shape[1]),qs=[0],spectra=['fitted_fm_embedding'],arms=['M','C'],groups=dict(zip(G,ns)),draws=499,bootstrap_resamples=2000,bootstrap_address=79,diagnostics=['standardized_between'],input_sha256=sha(EMB),protocol_sha256=sha(PROTOCOL),source_sha256={str(f.relative_to(R)).replace('\\','/'):sha(f) for f in (Path(__file__),Path(nc.__file__),Path(ov.__file__))},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    a.output.mkdir(parents=True,exist_ok=True)
    cp=a.output/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text(encoding='utf-8'))==cfg,'Configuration mismatch'
    else:
        cp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    jp=a.output/'records.jsonl'
    rows=[json.loads(l) for l in jp.read_text(encoding='utf-8').splitlines()] if jp.exists() else []
    done={(r['group'],r['index']) for r in rows}
    ts=[(root,g,i,cfg['draws']) for g in range(len(G)) for i in range(ns[g]) if (G[g],i) not in done]
    t0=time.time()
    with Pool(a.workers) as pool,jp.open('a',encoding='utf-8') as j:
        for r in pool.imap_unordered(one,ts,chunksize=1):
            j.write(json.dumps(r)+'\n');j.flush();rows.append(r)
            if len(rows)%100==0:
                print(f'{len(rows)}/{sum(ns)}; {time.time()-t0:.0f}s',flush=True)
    end=dict(n=len(rows),errors=sum('error' in r for r in rows),records_sha256=sha(jp),execution_wall_s=time.time()-t0)
    (a.output/'execution.json').write_text(json.dumps(end,indent=2),encoding='utf-8')
    print(json.dumps(end),flush=True)
    if end['errors']:
        raise RuntimeError('Errors retained; do not summarize as completed')

if __name__=='__main__':
    main()
