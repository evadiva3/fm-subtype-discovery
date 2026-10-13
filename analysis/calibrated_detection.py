import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[v]='1'
import argparse,hashlib,json,platform,sys,time,traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy,sklearn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'reproducibility/s16/modules'))
import angular_gof as gof
import nullcal_validation as nv
G=('calibration','null','leading2','leading4','minor2','minor4')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sd(root,*a):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(root,72,*a))).encode()).digest()[:16],'big')

def one(t):
    root,s,q,g,i,nd=t
    sid=0 if g==0 else 1
    ds=sd(root,0,s,sid,i)
    r=dict(spectrum=s,q=q,group=G[g],index=i,data_seed=str(ds))
    try:
        rng=np.random.default_rng(ds)
        z=rng.normal(size=(28,119))
        rot=np.linalg.qr(rng.normal(size=(119,119)))[0]
        sg=rng.choice([-1.,1.],size=28)
        lam=np.r_[1.,np.full(118,.01)] if s==0 else np.exp(-np.arange(119)/10.)
        ax=0 if g<4 else 59
        de=0. if g<2 else (2. if g%2==0 else 4.)
        x=(z*np.sqrt(lam))@rot.T+de*np.sqrt(lam[ax])*sg[:,None]*rot[:,ax]
        p=np.linalg.qr(np.random.default_rng(sd(root,1,0,0,0)).normal(size=(119,10)))[0][:,:q]
        xp=nv.normalize(nv.normalize(x)@p)
        cov=(p.T@rot*lam)@(rot.T@p)
        v=p.T@rot[:,ax]
        rs=sd(root,2,s,q,g,i)
        bs=sd(root,3,s,q,g,i)
        u=float(np.random.default_rng(sd(root,4,s,q,g,i)).uniform())
        r.update(delta=de,axis_index=ax,source_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),projected_sha256=hashlib.sha256(xp.astype('<f8').tobytes()).hexdigest(),projected_squared_half_separation=float(de**2*lam[ax]*v@np.linalg.solve(cov,v)),retained_axis_squared=float(v@v),reference_seed=str(rs),bootstrap_seed=str(bs),boundary_uniform=u)
        r.update(gof.evaluate(xp,draws=nd,seed=rs,bootstrap_seed=bs,auxiliary=200))
    except Exception:
        r['error']=traceback.format_exc()
    return r

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    root=2026092401 if a.pilot else 20260924
    ns=[5]*6 if a.pilot else [1000,1000,250,250,250,250]
    cfg=dict(root_seed=root,experiment=72,n=28,d=119,qs=[5,10],spectra=['two_block_1_0.01','exponential_tau10'],groups=dict(zip(G,ns)),draws=199,auxiliary=200,bootstrap_resamples=2000,protocol_sha256=sha(R/'docs/FOLLOWUP_PROTOCOL_20260924.md'),source_sha256={str(f.relative_to(R)).replace('\\','/'):sha(f) for f in [Path(__file__),*(R/'reproducibility/s16/modules').glob('*.py')]},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    a.output.mkdir(parents=True,exist_ok=True)
    cp=a.output/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text(encoding='utf-8'))==cfg,'Configuration mismatch'
    else:
        cp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    jp=a.output/'records.jsonl'
    rows=[json.loads(l) for l in jp.read_text(encoding='utf-8').splitlines()] if jp.exists() else []
    done={(r['spectrum'],r['q'],r['group'],r['index']) for r in rows}
    ts=[(root,s,q,g,i,199) for s in range(2) for q in (5,10) for g in range(6) for i in range(ns[g]) if (s,q,G[g],i) not in done]
    t0=time.time()
    with Pool(a.workers) as pool,jp.open('a',encoding='utf-8') as j:
        for r in pool.imap_unordered(one,ts,chunksize=1):
            j.write(json.dumps(r)+'\n')
            j.flush()
            rows.append(r)
            if len(rows)%50==0:
                print(f'{len(rows)}/{sum(ns)*4}; {time.time()-t0:.0f}s',flush=True)
    end=dict(n=len(rows),errors=sum('error' in r for r in rows),records_sha256=sha(jp),execution_wall_s=time.time()-t0)
    (a.output/'execution.json').write_text(json.dumps(end,indent=2),encoding='utf-8')
    print(json.dumps(end),flush=True)
    if end['errors']:
        raise RuntimeError('Errors retained; do not summarize as completed')

if __name__=='__main__':
    main()
